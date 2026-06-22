import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/mistral3
// Note: Mistral3 reuses the vision model from Pixtral

// perf: fuse silu(gate) * up into a single Metal kernel via MLX.compile,
// mirroring the Qwen35 SwiGLU fusion (commit c687780).
private let compiledSwiglu: @Sendable (MLXArray, MLXArray) -> MLXArray =
    MLX.compile(shapeless: true) { gate, up in silu(gate) * up }

// MARK: - Configuration

// Re-export PixtralVisionConfiguration for Mistral3 use
public typealias Mistral3VisionConfiguration = PixtralVisionConfiguration

// MARK: - Text Configuration

public struct Mistral3VLMTextConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let rmsNormEps: Float
    public let vocabSize: Int

    public var headDim: Int? { _headDim }
    public var maxPositionEmbeddings: Int? { _maxPositionEmbeddings }
    public var numKeyValueHeads: Int { _numKeyValueHeads ?? numAttentionHeads }
    public var ropeTheta: Float { _ropeTheta ?? 1_000_000_000 }
    public var ropeParameters: [String: StringOrNumber]? { _ropeParameters }
    public var ropeTraditional: Bool { _ropeTraditional ?? false }
    public var ropeScaling: [String: StringOrNumber]? { _ropeScaling }
    public var tieWordEmbeddings: Bool { _tieWordEmbeddings ?? false }
    public var layerTypes: [String]? { _layerTypes }
    public var slidingWindow: Int? { _slidingWindow }
    public var useQkNorm: Bool { _useQkNorm ?? false }

    private let _headDim: Int?
    private let _maxPositionEmbeddings: Int?
    private let _numKeyValueHeads: Int?
    private let _ropeTheta: Float?
    private let _ropeParameters: [String: StringOrNumber]?
    private let _ropeTraditional: Bool?
    private let _ropeScaling: [String: StringOrNumber]?
    private let _tieWordEmbeddings: Bool?
    private let _layerTypes: [String]?
    private let _slidingWindow: Int?
    private let _useQkNorm: Bool?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case _headDim = "head_dim"
        case _maxPositionEmbeddings = "max_position_embeddings"
        case _numKeyValueHeads = "num_key_value_heads"
        case _ropeTheta = "rope_theta"
        case _ropeParameters = "rope_parameters"
        case _ropeTraditional = "rope_traditional"
        case _ropeScaling = "rope_scaling"
        case _tieWordEmbeddings = "tie_word_embeddings"
        case _layerTypes = "layer_types"
        case _slidingWindow = "sliding_window"
        case _useQkNorm = "use_qk_norm"
    }
}

// MARK: - Model Configuration

public struct Mistral3VLMConfiguration: Codable, Sendable {
    public let textConfig: Mistral3VLMTextConfiguration
    public let visionConfig: Mistral3VisionConfiguration
    public let modelType: String

    public var ignoreIndex: Int { _ignoreIndex ?? -100 }
    public var imageTokenIndex: Int { _imageTokenIndex ?? _imageTokenId ?? 10 }
    public var visionFeatureSelectStrategy: String { _visionFeatureSelectStrategy ?? "full" }
    public var visionFeatureLayer: Int { _visionFeatureLayer ?? -1 }
    public var vocabSize: Int { _vocabSize ?? 32000 }
    public var spatialMergeSize: Int { _spatialMergeSize ?? 2 }
    public var multimodalProjectorBias: Bool { _multimodalProjectorBias ?? false }
    public var eosTokenId: [Int]? { _eosTokenId }

    private let _ignoreIndex: Int?
    private let _imageTokenIndex: Int?
    private let _imageTokenId: Int?
    private let _visionFeatureSelectStrategy: String?
    private let _visionFeatureLayer: Int?
    private let _vocabSize: Int?
    private let _spatialMergeSize: Int?
    private let _multimodalProjectorBias: Bool?
    private let _eosTokenId: [Int]?

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case modelType = "model_type"
        case _ignoreIndex = "ignore_index"
        case _imageTokenIndex = "image_token_index"
        case _imageTokenId = "image_token_id"
        case _visionFeatureSelectStrategy = "vision_feature_select_strategy"
        case _visionFeatureLayer = "vision_feature_layer"
        case _vocabSize = "vocab_size"
        case _spatialMergeSize = "spatial_merge_size"
        case _multimodalProjectorBias = "multimodal_projector_bias"
        case _eosTokenId = "eos_token_id"
    }
}

// MARK: - Unfold (im2col)

/// Extract sliding local blocks from a batched input tensor.
/// Equivalent to PyTorch's nn.functional.unfold / im2col operation.
private func unfold(
    _ input: MLXArray,
    kernelSize: Int,
    dilation: Int = 1,
    padding: Int = 0,
    stride: Int = 1
) -> MLXArray {
    var x = input
    let (batchSize, channels, height, width) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))

    // Add padding if needed
    if padding > 0 {
        x = MLX.padded(
            x,
            widths: [
                0,  // batch
                0,  // channels
                .init((padding, padding)),  // height
                .init((padding, padding)),  // width
            ])
    }

    let paddedH = height + 2 * padding
    let paddedW = width + 2 * padding

    // Calculate output dimensions
    let heightOut = (paddedH - dilation * (kernelSize - 1) - 1) / stride + 1
    let widthOut = (paddedW - dilation * (kernelSize - 1) - 1) / stride + 1

    // Extract blocks using array indexing
    var blocks: [MLXArray] = []

    for i in Swift.stride(from: 0, to: paddedH - kernelSize * dilation + 1, by: stride) {
        for j in Swift.stride(from: 0, to: paddedW - kernelSize * dilation + 1, by: stride) {
            var block: [MLXArray] = []
            for di in 0 ..< kernelSize {
                for dj in 0 ..< kernelSize {
                    let hIdx = i + di * dilation
                    let wIdx = j + dj * dilation
                    block.append(x[0..., 0..., hIdx, wIdx])
                }
            }
            // Stack the channel-blocks: (B, C, k*k)
            let stackedBlock = MLX.stacked(block, axis: 1).transposed(0, 2, 1)
            blocks.append(stackedBlock)
        }
    }

    // Stack all blocks: (B, C, k*k, L)
    let result = MLX.stacked(blocks, axis: -1)

    // Reshape to (B, C*k*k, L)
    return result.reshaped(batchSize, channels * kernelSize * kernelSize, heightOut * widthOut)
}

// MARK: - Mistral3 Patch Merger

private class Mistral3PatchMerger: Module {
    let spatialMergeSize: Int
    let patchSize: Int

    @ModuleInfo(key: "merging_layer") var mergingLayer: Linear

    init(_ config: Mistral3VLMConfiguration) {
        self.spatialMergeSize = config.spatialMergeSize
        self.patchSize = config.visionConfig.patchSize

        let hiddenSize = config.visionConfig.hiddenSize
        self._mergingLayer.wrappedValue = Linear(
            hiddenSize * spatialMergeSize * spatialMergeSize,
            hiddenSize,
            bias: false
        )
    }

    func callAsFunction(_ imageFeatures: MLXArray, imageSizes: [(Int, Int)]) -> MLXArray {
        // Convert image sizes to patch sizes
        let patchSizes = imageSizes.map { (h, w) in
            (h / patchSize, w / patchSize)
        }

        let tokensPerImage = patchSizes.map { $0.0 * $0.1 }
        let d = imageFeatures.dim(-1)
        var features = imageFeatures.asType(
            ProcessInfo.processInfo.environment["MLX_VLM_DTYPE"] == "float16"
                ? .float16 : .bfloat16)

        // Split the image features into chunks based on tokens per image
        var splitIndices: [Int] = []
        var currentIndex = 0
        for tokens in tokensPerImage.dropLast() {
            currentIndex += tokens
            splitIndices.append(currentIndex)
        }

        let chunks: [MLXArray]
        if splitIndices.isEmpty {
            chunks = [features[0, 0..., 0...]]
        } else {
            chunks = MLX.split(features[0], indices: splitIndices, axis: 0)
        }

        var permutedTensors: [MLXArray] = []

        for (imageIndex, imageTokens) in chunks.enumerated() {
            if imageTokens.dim(0) > 0 {
                let (h, w) = patchSizes[imageIndex]

                // Reshape to grid: (h, w, d) -> (1, d, h, w)
                let imageGrid = imageTokens.reshaped(h, w, d).transposed(2, 0, 1)[
                    .newAxis, 0..., 0..., 0...]

                // Apply unfold
                var grid = unfold(imageGrid, kernelSize: spatialMergeSize, stride: spatialMergeSize)

                // Reshape: (d * spatial_merge_size^2, -1).T
                grid = grid.reshaped(d * spatialMergeSize * spatialMergeSize, -1).transposed()
                permutedTensors.append(grid)
            }
        }

        features = MLX.concatenated(permutedTensors, axis: 0)
        features = mergingLayer(features)

        return features[.newAxis, 0..., 0...]
    }
}

// MARK: - Mistral3 MultiModal Projector

private class Mistral3MultiModalProjector: Module {
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "patch_merger") var patchMerger: Mistral3PatchMerger
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo var gelu: GELU
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ config: Mistral3VLMConfiguration) {
        self._norm.wrappedValue = RMSNorm(dimensions: config.visionConfig.hiddenSize)
        self._patchMerger.wrappedValue = Mistral3PatchMerger(config)
        self._linear1.wrappedValue = Linear(
            config.visionConfig.hiddenSize,
            config.textConfig.hiddenSize,
            bias: config.multimodalProjectorBias
        )
        self.gelu = GELU()
        self._linear2.wrappedValue = Linear(
            config.textConfig.hiddenSize,
            config.textConfig.hiddenSize,
            bias: config.multimodalProjectorBias
        )
    }

    func callAsFunction(_ x: MLXArray, imageSizes: [(Int, Int)]) -> MLXArray {
        var result = norm(x)
        result = patchMerger(result, imageSizes: imageSizes)
        result = linear1(result)
        result = gelu(result)
        result = linear2(result)
        return result
    }
}

// MARK: - Language Model Components

private enum Language {

    // MARK: Llama 4 Attention Scaling

    static func getLlama4AttentionScale(
        start: Int, stop: Int, beta: Float, maxPositionEmbeddings: Int
    ) -> MLXArray {
        let positions = MLXArray(start ..< stop).asType(.float32)
        let scaling = 1 + beta * MLX.log(1 + MLX.floor(positions / Float(maxPositionEmbeddings)))
        return expandedDimensions(scaling, axis: -1)
    }

    // MARK: Language Attention

    fileprivate class Attention: Module {
        let config: Mistral3VLMTextConfiguration
        let scale: Float
        let nHeads: Int
        let nKVHeads: Int
        let headDim: Int

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        let rope: RoPELayer

        init(_ config: Mistral3VLMTextConfiguration) {
            self.config = config

            let dim = config.hiddenSize
            self.nHeads = config.numAttentionHeads
            self.nKVHeads = config.numKeyValueHeads

            self.headDim = config.headDim ?? (config.hiddenSize / nHeads)
            self.scale = pow(Float(headDim), -0.5)

            self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
            self._wk.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
            self._wv.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
            self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

            // Initialize RoPE using rope_parameters - rope_theta is required like in Python
            guard let ropeParams = config.ropeParameters,
                let ropeTheta = ropeParams["rope_theta"]?.asFloat()
            else {
                fatalError("rope_parameters['rope_theta'] is required")
            }
            self.rope = initializeRope(
                dims: headDim,
                base: ropeTheta,
                traditional: false,
                scalingConfig: config.ropeParameters,
                maxPositionEmbeddings: config.maxPositionEmbeddings
            )
        }

        func callAsFunction(
            _ x: MLXArray,
            attentionScale: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

            let offset = cache?.offset ?? 0
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)

            queries = queries * attentionScale

            let output = attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values,
                cache: cache, scale: scale, mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return wo(output)
        }

        /// VisionZip/FastV-lite ranking signal: the attention the LAST query
        /// token pays to every position at this layer, averaged over heads.
        /// Single-query, so it needs no [L,L] materialization (flash SDPA hides
        /// the full weights — this recomputes just the one row we need).
        /// Returns shape [L]. Assumes B == 1 (single sequence).
        func lastTokenAttentionScores(_ x: MLXArray, attentionScale: MLXArray) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))
            var queries = wq(x)
            var keys = wk(x)
            queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
            queries = rope(queries, offset: 0)
            keys = rope(keys, offset: 0)
            queries = queries * attentionScale
            let repeats = nHeads / nKVHeads
            if repeats > 1 {
                keys = repeated(keys, count: repeats, axis: 1)
            }
            let qLast = queries[0..., 0..., (L - 1) ..< L, 0...]  // B, nHeads, 1, hd
            var scores = matmul(qLast, keys.transposed(0, 1, 3, 2)) * scale  // B, nHeads, 1, L
            scores = softmax(scores.asType(.float32), axis: -1)
            return scores.mean(axis: 1).reshaped([L])  // average heads -> [L]
        }

        /// VisionZip "dominant token" signal: the mean attention each position
        /// RECEIVES, averaged over heads and all causal query rows, at this layer
        /// (the `attn[:, :, img_pos].mean(0,1)` criterion from the Python probe).
        /// Chunked over queries so the full [L,L] map is never materialized.
        /// Returns shape [L]. Assumes B == 1.
        func receivedAttentionScores(
            _ x: MLXArray, attentionScale: MLXArray, additiveMask: MLXArray
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))
            var queries = wq(x).reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
            var keys = wk(x).reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
            queries = rope(queries, offset: 0)
            keys = rope(keys, offset: 0)
            queries = queries * attentionScale
            let repeats = nHeads / nKVHeads
            if repeats > 1 { keys = repeated(keys, count: repeats, axis: 1) }
            let keysT = keys.transposed(0, 1, 3, 2)  // B, nHeads, hd, L
            var acc: MLXArray? = nil  // per-key received-attention accumulator
            let block = 512
            var qs = 0
            while qs < L {
                let qe = min(qs + block, L)
                let qb = queries[0..., 0..., qs ..< qe, 0...]  // B, nHeads, blk, hd
                let sb = matmul(qb, keysT) * scale  // B, nHeads, blk, L
                let probs = softmax(sb.asType(.float32) + additiveMask[qs ..< qe, 0...], axis: -1)
                let contrib = probs.sum(axis: 2)  // B, nHeads, L
                acc = (acc == nil) ? contrib : acc! + contrib
                eval(acc!)
                qs = qe
            }
            return (acc!.mean(axis: 1) / Float(L)).reshaped([L])  // mean heads -> [L]
        }
    }

    // MARK: Language MLP

    fileprivate class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        init(_ config: Mistral3VLMTextConfiguration) {
            let dim = config.hiddenSize
            let hiddenDim = config.intermediateSize

            self._gate.wrappedValue = Linear(dim, hiddenDim, bias: false)
            self._down.wrappedValue = Linear(hiddenDim, dim, bias: false)
            self._up.wrappedValue = Linear(dim, hiddenDim, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(compiledSwiglu(gate(x), up(x)))
        }
    }

    // MARK: Language Transformer Block (for Ministral3 with attn_scale)

    fileprivate class TransformerBlock: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: MLP

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        let useSliding: Bool

        init(_ config: Mistral3VLMTextConfiguration, useSliding: Bool = false) {
            self.useSliding = useSliding
            self._attention.wrappedValue = Attention(config)
            self.mlp = MLP(config)
            self._inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        func callAsFunction(
            _ x: MLXArray,
            attentionScale: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?
        ) -> MLXArray {
            var r = attention(
                inputLayerNorm(x), attentionScale: attentionScale, mask: mask, cache: cache)
            let h = x + r
            r = mlp(postAttentionLayerNorm(h))
            return h + r
        }
    }

    // MARK: Ministral3 Model Inner (with sliding attention and llama4 scaling)

    fileprivate class Ministral3ModelInner: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

        let layers: [TransformerBlock]
        let norm: RMSNorm
        let config: Mistral3VLMTextConfiguration
        let layerTypes: [String]
        let slidingWindow: Int?
        let faIndex: Int
        let swaIndex: Int?

        init(_ config: Mistral3VLMTextConfiguration) {
            self.config = config
            self.slidingWindow = config.slidingWindow
            self.layerTypes =
                config.layerTypes
                ?? Array(repeating: "full_attention", count: config.numHiddenLayers)

            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabSize,
                dimensions: config.hiddenSize
            )

            self.layers = layerTypes.map { layerType in
                TransformerBlock(config, useSliding: layerType == "sliding_attention")
            }

            self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

            self.faIndex = layerTypes.firstIndex(of: "full_attention") ?? 0
            self.swaIndex = layers.firstIndex { $0.useSliding }
        }

        func callAsFunction(
            _ inputs: MLXArray,
            cache: [KVCache]?,
            inputsEmbeds: MLXArray? = nil
        ) -> MLXArray {
            var h: MLXArray
            if let inputsEmbeds {
                h = inputsEmbeds
            } else {
                h = embedTokens(inputs)
            }

            let cache = cache ?? []
            let offset = cache.first?.offset ?? 0

            let faMask = createAttentionMask(h: h, cache: cache[faIndex])

            var swaMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
            if let swaIndex, let slidingWindow, !cache.isEmpty {
                let t = h.dim(1)
                if t > 1 {
                    let swaOffset = min(slidingWindow, cache[swaIndex].offset)
                    swaMask = .array(
                        createCausalMask(n: t, offset: swaOffset, windowSize: slidingWindow))
                }
            }

            let beta = config.ropeParameters?["llama_4_scaling_beta"]?.asFloat() ?? 0.0
            let originalMaxPos =
                config.ropeParameters?["original_max_position_embeddings"]?.asInt()
                ?? config.maxPositionEmbeddings ?? 4096
            let attentionScale = getLlama4AttentionScale(
                start: offset,
                stop: offset + h.dim(1),  // Use h's length (embeddings), not inputs (token IDs)
                beta: beta,
                maxPositionEmbeddings: originalMaxPos
            ).asType(h.dtype)

            for (i, layer) in layers.enumerated() {
                let mask = layer.useSliding ? swaMask : faMask
                h = layer(
                    h, attentionScale: attentionScale, mask: mask,
                    cache: cache.isEmpty ? nil : cache[i])
            }

            return norm(h)
        }

        /// FastV-lite ranking: run the first `layerK` blocks over the full
        /// (un-pruned) embeddings, then return the last token's attention over
        /// every position at layer `layerK` (shape [L]). Used to rank visual
        /// tokens for pruning. Cost ≈ layerK/numLayers of one prefill.
        func imageRankScores(_ inputsEmbeds: MLXArray, layerK: Int) -> MLXArray {
            var h = inputsEmbeds
            let beta = config.ropeParameters?["llama_4_scaling_beta"]?.asFloat() ?? 0.0
            let originalMaxPos =
                config.ropeParameters?["original_max_position_embeddings"]?.asInt()
                ?? config.maxPositionEmbeddings ?? 4096
            let attentionScale = getLlama4AttentionScale(
                start: 0, stop: h.dim(1), beta: beta, maxPositionEmbeddings: originalMaxPos
            ).asType(h.dtype)
            let mask = createAttentionMask(h: h, cache: nil as KVCache?)
            let k = min(max(layerK, 0), layers.count - 1)
            for i in 0 ..< k {
                h = layers[i](h, attentionScale: attentionScale, mask: mask, cache: nil)
            }
            let xln = layers[k].inputLayerNorm(h)
            return layers[k].attention.lastTokenAttentionScores(xln, attentionScale: attentionScale)
        }

        /// VisionZip dominant-token ranking: run the first `layerK` blocks, then
        /// return the mean attention RECEIVED by every position (over heads + all
        /// causal queries) at layer `layerK`. Shape [L].
        func dominanceScores(_ inputsEmbeds: MLXArray, layerK: Int) -> MLXArray {
            var h = inputsEmbeds
            let beta = config.ropeParameters?["llama_4_scaling_beta"]?.asFloat() ?? 0.0
            let originalMaxPos =
                config.ropeParameters?["original_max_position_embeddings"]?.asInt()
                ?? config.maxPositionEmbeddings ?? 4096
            let attentionScale = getLlama4AttentionScale(
                start: 0, stop: h.dim(1), beta: beta, maxPositionEmbeddings: originalMaxPos
            ).asType(h.dtype)
            let mask = createAttentionMask(h: h, cache: nil as KVCache?)
            let k = min(max(layerK, 0), layers.count - 1)
            for i in 0 ..< k {
                h = layers[i](h, attentionScale: attentionScale, mask: mask, cache: nil)
            }
            let L = h.dim(1)
            let rinds = MLXArray(Int32(0) ..< Int32(L))
            let causal = rinds[0..., .newAxis] .>= rinds[.newAxis]  // [L,L] key<=query
            let additive = MLX.where(causal, MLXArray(Float(0)), MLXArray(Float(-1e9)))
            let xln = layers[k].inputLayerNorm(h)
            return layers[k].attention.receivedAttentionScores(
                xln, attentionScale: attentionScale, additiveMask: additive)
        }
    }

    // MARK: Language Model

    /// Language model that supports both ministral3 and mistral model types.
    /// For ministral3: uses sliding attention with llama4 attention scaling
    /// For mistral: uses standard attention with optional QK norm
    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        let config: Mistral3VLMTextConfiguration
        let modelType: String

        // Use ministral3 model as the primary implementation
        // It handles both cases: ministral3 with sliding attention, or standard with beta=0
        @ModuleInfo(key: "model") private var model: Ministral3ModelInner

        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        var kvHeads: [Int] {
            let layerTypes =
                config.layerTypes
                ?? Array(repeating: "full_attention", count: config.numHiddenLayers)
            return layerTypes.map { _ in config.numKeyValueHeads }
        }

        /// Access to embed_tokens
        var embedTokens: Embedding {
            model.embedTokens
        }

        /// Access to layers for LoRA
        var layers: [TransformerBlock] {
            model.layers
        }

        init(_ config: Mistral3VLMTextConfiguration) {
            self.config = config
            self.modelType = config.modelType

            // Ministral3ModelInner handles both model types:
            // - For ministral3: uses sliding attention and llama4 scaling from rope_parameters
            // - For mistral: when llama_4_scaling_beta is 0 or missing, attention_scale becomes 1.0
            //   and all layers use full attention (no layer_types means all "full_attention")
            self._model.wrappedValue = Ministral3ModelInner(config)

            if !config.tieWordEmbeddings {
                self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
            }
        }

        func callAsFunction(
            _ inputs: MLXArray,
            cache: [KVCache]?,
            inputsEmbeds: MLXArray? = nil
        ) -> MLXArray {
            var out = model(inputs, cache: cache, inputsEmbeds: inputsEmbeds)

            // perf: generation uses only the last position's logits — project that
            // token alone, not all L prefill positions, through the vocab-sized head.
            if out.dim(1) > 1 { out = out[0..., (out.dim(1) - 1) ..< out.dim(1), 0...] }
            if config.tieWordEmbeddings {
                out = embedTokens.asLinear(out)
            } else if let lmHead {
                out = lmHead(out)
            }
            return out
        }

        /// FastV-lite visual-token ranking (delegates to the inner model).
        func imageRankScores(_ inputsEmbeds: MLXArray, layerK: Int) -> MLXArray {
            model.imageRankScores(inputsEmbeds, layerK: layerK)
        }

        /// VisionZip dominant-token ranking (delegates to the inner model).
        func dominanceScores(_ inputsEmbeds: MLXArray, layerK: Int) -> MLXArray {
            model.dominanceScores(inputsEmbeds, layerK: layerK)
        }

        func newCache(parameters: GenerateParameters?) -> [KVCache] {
            let layerTypes =
                config.layerTypes
                ?? Array(repeating: "full_attention", count: config.numHiddenLayers)

            return layerTypes.map { layerType in
                if layerType == "sliding_attention", let slidingWindow = config.slidingWindow {
                    return RotatingKVCache(maxSize: slidingWindow)
                } else if let maxKVSize = parameters?.maxKVSize {
                    return RotatingKVCache(maxSize: maxKVSize, keep: 4)
                } else {
                    return KVCacheSimple()
                }
            }
        }
    }
}

// MARK: - Mistral3 VLM Model

public class Mistral3VLM: Module, VLMModel, KVCacheDimensionProvider {
    // Use PixtralVision.VisionModel from Pixtral.swift
    @ModuleInfo(key: "vision_tower") private var visionTower: PixtralVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Language.LanguageModel
    @ModuleInfo(key: "multi_modal_projector") private var multiModalProjector:
        Mistral3MultiModalProjector

    public let config: Mistral3VLMConfiguration
    let visionFeatureLayer: Int

    public var vocabularySize: Int { config.vocabSize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public init(_ config: Mistral3VLMConfiguration) {
        self.config = config
        self.visionFeatureLayer = config.visionFeatureLayer

        self._visionTower.wrappedValue = PixtralVision.VisionModel(config.visionConfig)
        self._languageModel.wrappedValue = Language.LanguageModel(config.textConfig)
        self._multiModalProjector.wrappedValue = Mistral3MultiModalProjector(config)
    }

    private func getInputEmbeddings(
        inputIds: MLXArray?,
        pixelValues: MLXArray?,
        imageSizes: [(Int, Int)]?
    ) -> MLXArray {
        guard var pixelValues, let imageSizes else {
            guard let inputIds else {
                fatalError("Either inputIds or pixelValues must be provided")
            }
            return languageModel.embedTokens(inputIds)
        }

        guard let inputIds else {
            fatalError("inputIds required when pixelValues provided")
        }

        let inputsEmbeds = languageModel.embedTokens(inputIds)

        // Handle 3D pixel values (missing batch dimension)
        if pixelValues.ndim == 3 {
            pixelValues = pixelValues.expandedDimensions(axis: 0)
        }

        let prof = Self.profileEnabled
        func ms() -> Double { Date().timeIntervalSince1970 * 1000 }
        let t0 = ms()
        func selLayer(_ hs: [MLXArray]) -> MLXArray {
            let i = visionFeatureLayer < 0 ? hs.count + visionFeatureLayer : visionFeatureLayer
            return hs[i]
        }

        let imageFeatures: MLXArray
        let t1: Double
        if ProcessInfo.processInfo.environment["VLCACHE_MODE"] == "cropencode",
            let cached = PixtralVision.current.mergeTokens,
            let bboxStr = ProcessInfo.processInfo.environment["VLCACHE_BBOX"] {
            // VLCache crop-encode: encode ONLY the changed region (with full-frame
            // position offset) and splice into cached full-frame tokens. The encoder
            // runs on the crop, not the full frame — the real timing win.
            let H = pixelValues.dim(2), W = pixelValues.dim(3)
            let p = bboxStr.split(separator: ",").compactMap { Double($0) }
            let x0 = (Int(Double(W) * p[0]) / 28) * 28
            let y0 = (Int(Double(H) * p[1]) / 28) * 28
            let x1 = min(W, ((Int(Double(W) * p[2]) + 27) / 28) * 28)
            let y1 = min(H, ((Int(Double(H) * p[3]) + 27) / 28) * 28)
            let cropPix = pixelValues[0..., 0..., y0 ..< y1, x0 ..< x1]
            setenv("VISION_POS_OFFSET", "\(y0 / 14),\(x0 / 14)", 1)
            let (_, _, cropHidden) = visionTower(cropPix.transposed(0, 2, 3, 1), outputHiddenStates: true)
            unsetenv("VISION_POS_OFFSET")
            let cropSel = selLayer(cropHidden!)
            if prof { eval(cropSel) }
            t1 = ms()
            let cropMerged = multiModalProjector(cropSel, imageSizes: [(y1 - y0, x1 - x0)])
            imageFeatures = Self.spliceCrop(
                cached: cached, crop: cropMerged,
                fullGridW: W / 28, fullGridH: H / 28,
                gr0: y0 / 28, gc0: x0 / 28,
                cropGridW: (x1 - x0) / 28, cropGridH: (y1 - y0) / 28)
        } else {
            // Process through vision tower (reuses Pixtral vision model)
            let (_, _, hiddenStates) = visionTower(
                pixelValues.transposed(0, 2, 3, 1), outputHiddenStates: true)
            guard let hiddenStates else { fatalError("Vision model must return hidden states") }
            let selectedFeatures = selLayer(hiddenStates)
            if prof { eval(selectedFeatures) }
            t1 = ms()
            // VLCache Option-2 splice (validation path): cached static + fresh changed.
            imageFeatures = applyVLCacheSplice(
                multiModalProjector(selectedFeatures, imageSizes: imageSizes), imageSizes: imageSizes)
        }

        // Merge embeddings
        let fullEmbeds = mergeInputIdsWithImageFeatures(
            imageTokenIndex: config.imageTokenIndex,
            imageFeatures: imageFeatures,
            inputsEmbeds: inputsEmbeds,
            inputIds: inputIds
        )
        if prof { eval(fullEmbeds) }
        let t2 = ms()

        // VisionZip / FastV-lite: attention-ranked visual-token pruning (env-gated
        // via VISIONZIP_KEEP). The LM ignores inputIds when inputsEmbeds is given,
        // so pruning the embeddings alone shortens the prefill; KV-cache length and
        // RoPE positions follow the pruned sequence automatically.
        guard let keep = Self.visionZipKeep, keep < 1.0, inputIds.dim(0) == 1 else {
            Self.profileTimes = (vision: t1 - t0, merge: t2 - t1, rank: 0)
            return fullEmbeds
        }
        let pruned = pruneVisualTokens(fullEmbeds: fullEmbeds, inputIds: inputIds, keep: keep)
        if prof { eval(pruned) }
        Self.profileTimes = (vision: t1 - t0, merge: t2 - t1, rank: ms() - t2)
        return pruned
    }

    /// Set the active VLCache handle for the next inference(s). Engine owns one per
    /// context and calls this before generate; nil uses the shared default.
    public func setVLCache(_ handle: VLCacheHandle?) { PixtralVision.activeHandle = handle }

    /// Splice freshly-encoded crop tokens into the cached full-frame token grid by
    /// position. Changed-region grid cells take the crop tokens (encoded with the
    /// matching full-frame position offset); everything else reuses the cache.
    private static func spliceCrop(
        cached: MLXArray, crop: MLXArray,
        fullGridW: Int, fullGridH: Int, gr0: Int, gc0: Int, cropGridW: Int, cropGridH: Int
    ) -> MLXArray {
        let nFull = fullGridW * fullGridH
        guard cached.dim(1) == nFull else { return cached }  // grid mismatch -> safe fallback
        var gi = [Int32](repeating: 0, count: nFull)
        for t in 0 ..< nFull {
            let r = t / fullGridW, c = t % fullGridW
            if r >= gr0 && r < gr0 + cropGridH && c >= gc0 && c < gc0 + cropGridW {
                gi[t] = Int32(nFull + (r - gr0) * cropGridW + (c - gc0))
            } else {
                gi[t] = Int32(t)
            }
        }
        if ProcessInfo.processInfo.environment["VISIONZIP_LOG"] != nil {
            print("[vlcache] cropencode: full \(fullGridW)x\(fullGridH)=\(nFull), crop \(cropGridW)x\(cropGridH)=\(crop.dim(1)) at (\(gr0),\(gc0)); reused \(nFull - cropGridW * cropGridH)")
        }
        let combined = MLX.concatenated([cached[0], crop[0]], axis: 0)
        return MLX.take(combined, MLXArray(gi), axis: 0)[.newAxis, 0..., 0...]
    }

    /// VLCache Option-2 splice (env-gated, token-level reuse).
    ///   VLCACHE_MODE=record  -> cache this frame's post-merge image tokens.
    ///   VLCACHE_MODE=splice  -> keep cached tokens OUTSIDE the changed bbox
    ///       (VLCACHE_BBOX="fx0,fy0,fx1,fy1" in [0,1]); use this frame's tokens inside.
    /// Reuses cached encoded tokens for the static region, validating whether a
    /// spliced (cached static + fresh changed) token sequence yields correct OCR.
    private func applyVLCacheSplice(_ feats: MLXArray, imageSizes: [(Int, Int)]) -> MLXArray {
        guard let mode = ProcessInfo.processInfo.environment["VLCACHE_MODE"] else { return feats }
        if mode == "record" {
            eval(feats)
            PixtralVision.current.mergeTokens = feats
            return feats
        }
        guard mode == "splice", let cached = PixtralVision.current.mergeTokens,
            cached.dim(1) == feats.dim(1), let (h, w) = imageSizes.first
        else { return feats }
        let n = feats.dim(1)
        // Recover the merged grid (gridW × gridH = n, aspect ≈ w/h, row-major).
        let aspect = Double(w) / Double(h)
        var gridW = max(1, Int((Double(n) * aspect).squareRoot().rounded()))
        while gridW > 1 && n % gridW != 0 { gridW -= 1 }
        let gridH = n / gridW
        let p = (ProcessInfo.processInfo.environment["VLCACHE_BBOX"] ?? "0,0,1,1")
            .split(separator: ",").compactMap { Double($0) }
        guard p.count == 4 else { return feats }
        let (fx0, fy0, fx1, fy1) = (p[0], p[1], p[2], p[3])
        // mask[t] = 1.0 -> reuse cached (token OUTSIDE the changed bbox), 0.0 -> fresh.
        var m = [Float](repeating: 0, count: n)
        for t in 0 ..< n {
            let r = Double(t / gridW) / Double(gridH)
            let c = Double(t % gridW) / Double(gridW)
            let inside = c >= fx0 && c <= fx1 && r >= fy0 && r <= fy1
            m[t] = inside ? 0.0 : 1.0
        }
        let reused = m.reduce(0) { $0 + ($1 > 0.5 ? 1 : 0) }
        let mask = MLXArray(m).reshaped([1, n, 1]).asType(feats.dtype)
        if ProcessInfo.processInfo.environment["VISIONZIP_LOG"] != nil {
            print("[vlcache] splice grid=\(gridW)x\(gridH) n=\(n): reused \(reused) cached, \(n - reused) fresh")
        }
        return mask * cached.asType(feats.dtype) + (MLXArray(Float(1)).asType(feats.dtype) - mask) * feats
    }

    /// VISIONZIP_KEEP: fraction of visual tokens to retain (e.g. 0.2 = keep 20%).
    private static var visionZipKeep: Float? {
        guard let s = ProcessInfo.processInfo.environment["VISIONZIP_KEEP"],
            let v = Float(s) else { return nil }
        return v
    }
    /// VISIONZIP_LAYER: LM layer whose last-token attention ranks the tokens.
    private static var visionZipLayer: Int {
        ProcessInfo.processInfo.environment["VISIONZIP_LAYER"].flatMap { Int($0) } ?? 3
    }
    /// VISIONZIP_PROFILE: when set, getInputEmbeddings/prepare time each phase
    /// (vision encode / merge / ranking / LM prefill) with eval() barriers.
    private static var profileEnabled: Bool {
        ProcessInfo.processInfo.environment["VISIONZIP_PROFILE"] != nil
    }
    /// Sub-phase times (ms) stashed by getInputEmbeddings for prepare() to print.
    nonisolated(unsafe) static var profileTimes: (vision: Double, merge: Double, rank: Double) = (0, 0, 0)

    /// Keep the top-`keep` fraction of image tokens by FastV attention score,
    /// drop the rest, and keep every non-image position. Returns the pruned
    /// embeddings (shorter sequence). One CPU sync for the ranking — prefill only.
    private func pruneVisualTokens(fullEmbeds: MLXArray, inputIds: MLXArray, keep: Float)
        -> MLXArray
    {
        let L = fullEmbeds.dim(1)
        let imgTok = Int32(config.imageTokenIndex)
        let idsRow = inputIds[0].asType(.int32).asArray(Int32.self)
        var imgPos = [Int]()
        for (i, t) in idsRow.enumerated() where t == imgTok { imgPos.append(i) }
        let n = imgPos.count
        let k = max(1, Int((Float(n) * keep).rounded()))
        if n == 0 || k >= n { return fullEmbeds }

        // VISIONZIP_STRATEGY: "visionzip" (dominant tokens = attention received,
        // default), "fastv" (last-token attention), or "uniform" (evenly-spaced
        // control). VisionZip is the method matched to the Python probe.
        let strategy = ProcessInfo.processInfo.environment["VISIONZIP_STRATEGY"] ?? "visionzip"
        let layerK = Self.visionZipLayer
        let keepImg: Set<Int>
        switch strategy {
        case "uniform":
            let step = Double(n) / Double(k)
            keepImg = Set((0 ..< k).map { imgPos[min(n - 1, Int(Double($0) * step))] })
        case "fastv":
            let scores = languageModel.imageRankScores(fullEmbeds, layerK: layerK)
                .asArray(Float.self)
            keepImg = Set(imgPos.sorted { scores[$0] > scores[$1] }.prefix(k))
        default:  // "visionzip"
            let scores = languageModel.dominanceScores(fullEmbeds, layerK: layerK)
                .asArray(Float.self)
            if ProcessInfo.processInfo.environment["VISIONZIP_DIVERSITY"] != nil {
                keepImg = diverseKeep(fullEmbeds: fullEmbeds, imgPos: imgPos, scores: scores, k: k)
            } else {
                keepImg = Set(imgPos.sorted { scores[$0] > scores[$1] }.prefix(k))
            }
        }

        var keepIdx = [Int32]()
        keepIdx.reserveCapacity(L - (n - k))
        for i in 0 ..< L where idsRow[i] != imgTok || keepImg.contains(i) {
            keepIdx.append(Int32(i))
        }
        let pruned = MLX.take(fullEmbeds, MLXArray(keepIdx), axis: 1)
        if ProcessInfo.processInfo.environment["VISIONZIP_LOG"] != nil {
            let div = ProcessInfo.processInfo.environment["VISIONZIP_DIVERSITY"] != nil ? "+div" : ""
            print("[visionzip] keep=\(keep) strat=\(strategy)\(div) layer=\(layerK): \(n) img -> \(k); seq \(L) -> \(pruned.dim(1))")
        }
        return pruned
    }

    /// VisPruner-style diversity selection: from the top dominant candidates (by
    /// attention received), greedily pick `k` via farthest-point on cosine similarity
    /// of their embeddings, so the kept set spreads across the image instead of
    /// clustering on one salient region. CPU greedy over ~2k candidates — prefill only.
    private func diverseKeep(fullEmbeds: MLXArray, imgPos: [Int], scores: [Float], k: Int)
        -> Set<Int>
    {
        let n = imgPos.count
        let mult = ProcessInfo.processInfo.environment["VISIONZIP_DIVERSITY_MULT"]
            .flatMap { Double($0) } ?? 2.0
        let candCount = min(n, max(k + 1, Int(Double(k) * mult)))
        let cand = Array(imgPos.sorted { scores[$0] > scores[$1] }.prefix(candCount))
        // normalized candidate embeddings -> cosine similarity matrix
        let E = MLX.take(fullEmbeds[0], MLXArray(cand.map { Int32($0) }), axis: 0).asType(.float32)
        let En = E / sqrt((E * E).sum(axis: -1, keepDims: true) + 1e-6)
        let sim = matmul(En, En.transposed(1, 0)).asArray(Float.self)  // [candCount*candCount]
        // greedy farthest-point, seeded with the most dominant candidate (cand[0])
        var selected = Set<Int>([0])
        var order = [0]
        var maxSim = (0 ..< candCount).map { sim[$0 * candCount + 0] }
        while order.count < k {
            var best = -1
            var bestVal = Float.greatestFiniteMagnitude
            for j in 0 ..< candCount where !selected.contains(j) {
                if maxSim[j] < bestVal { bestVal = maxSim[j]; best = j }
            }
            if best < 0 { break }
            selected.insert(best); order.append(best)
            for j in 0 ..< candCount {
                let s = sim[j * candCount + best]
                if s > maxSim[j] { maxSim[j] = s }
            }
        }
        return Set(order.map { cand[$0] })
    }

    private func mergeInputIdsWithImageFeatures(
        imageTokenIndex: Int,
        imageFeatures: MLXArray,
        inputsEmbeds: MLXArray,
        inputIds: MLXArray
    ) -> MLXArray {
        // perf: pure-GPU masked scatter, mirroring the Qwen35 merge (commit
        // f69f820). The previous path pulled inputIds to the CPU via
        // `asArray` (a GPU→CPU sync every prefill), then split the image
        // features into one slice *per patch* and concatenated — O(patches)
        // Swift-side ops. The cumsum trick stays entirely on the GPU:
        //   1. mask = (inputIds == imageToken), broadcast to the embed shape.
        //   2. cumsum(mask)-1 gives, at each True position, its index into the
        //      flattened image features (a stale index at False positions).
        //   3. `% featuresSize` keeps the gather in bounds on False positions.
        //   4. where(mask, gathered, embeds) selects the right value per slot.
        // We trust the processor to emit exactly numImagePatches image tokens
        // (the Python mlx-vlm impl makes the same assumption); a mismatch
        // surfaces downstream rather than via a CPU-sync count here.
        let mask = (inputIds .== MLXArray(Int32(imageTokenIndex)))
        let specialMask = expandedDimensions(mask, axis: -1)
        let maskExpanded = broadcast(specialMask, to: inputsEmbeds.shape)

        let originalShape = inputsEmbeds.shape
        let flattenedEmbeds = inputsEmbeds.flattened()
        let flattenedFeatures = imageFeatures.flattened()
        let flattenedMask = maskExpanded.flattened()

        let maskInt = flattenedMask.asType(.int32)
        let positionIndex = (cumsum(maskInt) - MLXArray(Int32(1))).asType(.int32)
        let featuresSize = MLXArray(Int32(flattenedFeatures.size))
        let aligned = MLX.take(flattenedFeatures, positionIndex % featuresSize, axis: 0)
        let resultFlat = MLX.where(flattenedMask, aligned, flattenedEmbeds)

        return resultFlat.reshaped(originalShape)
    }

    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let inputIds = input.text.tokens
        let pixelValues = input.image?.pixels

        // Extract image sizes from frames or fall back to config defaults
        let imageSizes: [(Int, Int)]?
        if let frames = input.image?.frames {
            imageSizes = frames.map { ($0.h, $0.w) }
        } else if pixelValues != nil {
            imageSizes = [(config.visionConfig.imageSize, config.visionConfig.imageSize)]
        } else {
            imageSizes = nil
        }

        let embeddings = getInputEmbeddings(
            inputIds: inputIds,
            pixelValues: pixelValues,
            imageSizes: imageSizes
        )

        if Mistral3PrefixStore.enabled,
            let cached = prefixCachedPrefill(
                inputIds: inputIds, embeddings: embeddings, cache: cache)
        {
            return cached
        }

        let prof = Self.profileEnabled
        let tLm0 = Date().timeIntervalSince1970 * 1000
        let logits = languageModel(inputIds, cache: cache, inputsEmbeds: embeddings)
        if prof {
            eval(logits)
            let lm = Date().timeIntervalSince1970 * 1000 - tLm0
            let p = Self.profileTimes
            print(String(
                format:
                    "[breakdown] vision=%.0fms  merge=%.0fms  rank=%.0fms  lm_prefill=%.0fms  | seq=%d→%d",
                p.vision, p.merge, p.rank, lm, inputIds.dim(1), embeddings.dim(1)))
        }
        return .logits(.init(logits: logits))
    }

    /// Prompt-prefix KV cache (PREFIX_CACHE=1). The screen-analysis prompt's
    /// instruction block is identical across requests; with MISTRAL3_TEXT_FIRST=1
    /// those tokens precede the image, so their KV is reusable. The static
    /// boundary is discovered as the longest common token prefix with the
    /// previous request (capped at the first image token), so the per-request
    /// dynamic context (app name, window title) never enters the snapshot.
    /// Request 1: full prefill (learns tokens). Request 2: two-phase prefill,
    /// snapshot at the boundary. Request 3+: restore snapshot, prefill suffix only.
    private func prefixCachedPrefill(
        inputIds: MLXArray, embeddings: MLXArray, cache: [KVCache]
    ) -> PrepareResult? {
        // Snapshot/restore relies on KVCacheSimple semantics (state setter
        // derives offset from the array). Bail out for any other cache type
        // (e.g. RotatingKVCache when maxKVSize is set).
        guard cache.allSatisfy({ $0 is KVCacheSimple }) else { return nil }
        let store = Mistral3PrefixStore.shared
        let flat: [Int32] = inputIds[0].asArray(Int32.self)
        let n = flat.count
        let log = ProcessInfo.processInfo.environment["PREFIX_CACHE_LOG"] == "1"

        // Reuse path: current tokens start with the snapshotted prefix.
        let k = store.tokens.count
        if k >= 128, n > k, !store.caches.isEmpty, Array(flat[0 ..< k]) == store.tokens {
            for i in cache.indices {
                var c = cache[i]
                c.state = store.caches[i].state
            }
            let logits = languageModel(
                inputIds[0..., k ..< n], cache: cache,
                inputsEmbeds: embeddings[0..., k ..< n, 0...])
            if log { print("[prefixcache] HIT reused=\(k) prefilled=\(n - k)") }
            store.lastTokens = flat
            return .logits(.init(logits: logits))
        }

        // Record path: boundary = LCP with previous request, capped at first
        // image token (image features are per-request, never cacheable).
        defer { store.lastTokens = flat }
        let imgId = Int32(config.imageTokenIndex)
        let firstImage = flat.firstIndex(of: imgId) ?? n
        guard let boundary = Self.recordBoundary(
            flat: flat, lastTokens: store.lastTokens, firstImage: firstImage)
        else {
            if log { print("[prefixcache] MISS no usable boundary (img=\(firstImage) n=\(n))") }
            return nil  // fall through to the standard single-shot prefill
        }
        _ = languageModel(
            inputIds[0..., 0 ..< boundary], cache: cache,
            inputsEmbeds: embeddings[0..., 0 ..< boundary, 0...])
        store.tokens = Array(flat[0 ..< boundary])
        store.caches = cache.map { $0.copy() }
        let logits = languageModel(
            inputIds[0..., boundary ..< n], cache: cache,
            inputsEmbeds: embeddings[0..., boundary ..< n, 0...])
        if log { print("[prefixcache] RECORD snapshot=\(boundary) prefilled=\(n - boundary)") }
        return .logits(.init(logits: logits))
    }

    /// Boundary for the prefix-cache record path: the longest common token
    /// prefix with the previous request, capped at the first image token
    /// (image features are per-request, never cacheable). Pure so the guard is
    /// unit-testable without loading model weights (see Mistral3PrefixCacheTests).
    ///
    /// Returns nil — caller falls back to a normal single-shot prefill — when
    /// the prefix is too short to be worth caching (< `minPrefix`) OR when it
    /// would consume the whole sequence (`boundary == n`). The latter leaves an
    /// empty suffix slice `[boundary..<n]`; prefilling a zero-length sequence
    /// feeds an empty array into attention and crashes MLX's `reshape`
    /// ("Cannot infer the shape of an empty array"), killing the helper. This
    /// happens for text-only requests (no image token → `firstImage == n`)
    /// whose tokens are identical to / a prefix of the previous request's.
    /// Reported as Sentry RECALL-ADHD-3A.
    static func recordBoundary(
        flat: [Int32], lastTokens: [Int32], firstImage: Int, minPrefix: Int = 128
    ) -> Int? {
        let n = flat.count
        var lcp = 0
        while lcp < lastTokens.count && lcp < n && lastTokens[lcp] == flat[lcp] { lcp += 1 }
        let boundary = min(lcp, firstImage)
        guard boundary >= minPrefix, boundary < n else { return nil }
        return boundary
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights: [String: MLXArray] = [:]

        for (key, value) in weights {
            var newKey = key

            // Transform keys to match model structure
            // Vision tower keys: vision_tower.X -> vision_tower.vision_model.X (for pixtral structure)
            if key.contains("vision_tower") && !key.contains("vision_model") {
                if key.contains("transformer") || key.contains("patch_conv")
                    || key.contains("ln_pre")
                {
                    newKey = key.replacingOccurrences(
                        of: "vision_tower", with: "vision_tower.vision_model")
                }
            } else if key.contains("vision_encoder") && !key.contains("vision_tower") {
                // Alternative key format: model.vision_encoder.X -> vision_tower.vision_model.X
                if key.contains("transformer") || key.contains("patch_conv")
                    || key.contains("ln_pre")
                {
                    newKey = key.replacingOccurrences(
                        of: "model.vision_encoder", with: "vision_tower.vision_model")
                }
            } else if key.contains("model.language_model") && !key.contains("language_model.model")
            {
                newKey = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            } else if key.contains("lm_head") && !key.contains("language_model") {
                newKey = key.replacingOccurrences(of: "lm_head", with: "language_model.lm_head")
            } else if key.contains("model.vision_projection") {
                newKey = key.replacingOccurrences(
                    of: "model.vision_projection", with: "multi_modal_projector")
            }

            // Skip rotary embeddings
            if newKey.contains("self_attn.rotary_emb.inv_freq") {
                continue
            }

            // Handle weight scale patterns
            if newKey.contains("weight_scale_inv") {
                let scaleInv = value
                let weightKey = newKey.replacingOccurrences(of: "_scale_inv", with: "")
                if let weight = weights[key.replacingOccurrences(of: "_scale_inv", with: "")] {
                    newWeights[weightKey] = weight * scaleInv
                }
            } else if newKey.contains("activation_scale") {
                continue
            } else if newWeights[newKey] == nil {
                newWeights[newKey] = value
            }
        }

        return newWeights
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }
}

// MARK: - LoRA Support

extension Mistral3VLM: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.layers
    }
}

// MARK: - Processor Configuration

public struct Mistral3VLMProcessorConfiguration: Codable, Sendable {
    public let imageProcessor: ImageProcessorConfig
    public let imageToken: String
    public let imageBreakToken: String?
    public let imageEndToken: String?
    public let patchSize: Int
    public let spatialMergeSize: Int?

    public struct ImageProcessorConfig: Codable, Sendable {
        public let imageMean: [CGFloat]
        public let imageStd: [CGFloat]
        public let size: ProcessorSize
        public let patchSize: Int
        public let doNormalize: Bool?
        public let doRescale: Bool?
        public let doResize: Bool?
        public let rescaleFactor: Float?

        public struct ProcessorSize: Codable, Sendable {
            public let width: Int?
            public let height: Int?
            public let longestEdge: Int?

            enum CodingKeys: String, CodingKey {
                case width
                case height
                case longestEdge = "longest_edge"
            }
        }

        public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
            (imageMean[0], imageMean[1], imageMean[2])
        }

        public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
            (imageStd[0], imageStd[1], imageStd[2])
        }

        enum CodingKeys: String, CodingKey {
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case size
            case patchSize = "patch_size"
            case doNormalize = "do_normalize"
            case doRescale = "do_rescale"
            case doResize = "do_resize"
            case rescaleFactor = "rescale_factor"
        }
    }

    enum CodingKeys: String, CodingKey {
        case imageProcessor = "image_processor"
        case imageToken = "image_token"
        case imageBreakToken = "image_break_token"
        case imageEndToken = "image_end_token"
        case patchSize = "patch_size"
        case spatialMergeSize = "spatial_merge_size"
    }
}

// MARK: - Message Generator for Mistral3 VLM

/// Message generator for Mistral3 VLM that creates structured messages with image placeholders
/// Process-global snapshot for the prompt-prefix KV cache (PREFIX_CACHE=1).
/// Single-slot; the helper serializes requests so no locking is needed.
final class Mistral3PrefixStore: @unchecked Sendable {
    nonisolated(unsafe) static let shared = Mistral3PrefixStore()
    static let enabled = ProcessInfo.processInfo.environment["PREFIX_CACHE"] == "1"
    var tokens: [Int32] = []
    var lastTokens: [Int32] = []
    var caches: [any KVCache] = []
}

public struct Mistral3MessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> Message {
        // For Mistral3 VLM, images come before text in the content.
        // MISTRAL3_TEXT_FIRST=1 flips to text-before-image so the constant
        // instruction tokens form a cacheable prefix (see PREFIX_CACHE).
        let images: [[String: any Sendable]] = message.images.map { _ in ["type": "image"] }
        let text: [[String: any Sendable]] = [["type": "text", "text": message.content]]
        let textFirst = ProcessInfo.processInfo.environment["MISTRAL3_TEXT_FIRST"] == "1"
        return [
            "role": message.role.rawValue,
            "content": textFirst ? text + images : images + text,
        ]
    }
}

// MARK: - Processor

public struct Mistral3VLMProcessor: UserInputProcessor {
    private let config: Mistral3VLMProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let imageToken: String
    private let imageTokenId: Int

    private struct PreprocessResult {
        let pixels: MLXArray  // BCHW
        let frames: [THW]
        let numImageTokens: Int
    }

    public init(_ config: Mistral3VLMProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        self.imageToken = config.imageToken
        // Get image token ID from tokenizer, fallback to 10 (default for Mistral3)
        if let vocabTokenId = tokenizer.convertTokenToId(config.imageToken) {
            self.imageTokenId = vocabTokenId
        } else {
            self.imageTokenId = 10
        }
    }

    private func preprocessImage(
        _ image: CIImage,
        processing: UserInput.Processing?,
        patchSize: Int,
        spatialMergeSize: Int,
        longestEdge: Int?
    ) throws -> PreprocessResult {
        var image = MediaProcessing.inSRGBToneCurveSpace(image)
        image = MediaProcessing.apply(image, processing: processing)

        // Honor the model's configured longest_edge (Pixtral native max = 1540).
        // Was hardcoded to patchSize*24 = 336px, which crushed screenshot detail.
        let maxVisionEdge = 1540
        let targetEdge = min(longestEdge ?? maxVisionEdge, maxVisionEdge)

        let originalSize = image.extent.size
        let scale = min(CGFloat(targetEdge) / max(originalSize.width, originalSize.height), 1.0)
        let newWidth = max(1, Int((originalSize.width * scale).rounded()))
        let newHeight = max(1, Int((originalSize.height * scale).rounded()))

        // Round to patch size multiples for padding
        let paddedWidth = ((newWidth + patchSize - 1) / patchSize) * patchSize
        let paddedHeight = ((newHeight + patchSize - 1) / patchSize) * patchSize

        // Resize
        image = MediaProcessing.resampleBicubic(
            image,
            to: CGSize(width: newWidth, height: newHeight)
        )

        // Pad to patch boundaries (bottom-right padding)
        if newWidth != paddedWidth || newHeight != paddedHeight {
            let background = CIImage(color: .black).cropped(
                to: CGRect(x: 0, y: 0, width: paddedWidth, height: paddedHeight))
            let tx = 0.0
            let ty = CGFloat(paddedHeight - newHeight)
            let transformed = image.transformed(by: CGAffineTransform(translationX: tx, y: ty))
            image = transformed.composited(over: background)
        }

        image = MediaProcessing.normalize(
            image,
            mean: config.imageProcessor.imageMeanTuple,
            std: config.imageProcessor.imageStdTuple
        )

        var pixels = MediaProcessing.asMLXArray(image)

        if pixels.ndim == 2 {
            pixels = pixels.expandedDimensions(axis: -1)
        }
        if pixels.ndim == 3 {
            pixels = pixels.expandedDimensions(axis: 0)
        }
        // Convert to BCHW format for vision model
        if pixels.dim(-1) == 3 {
            pixels = pixels.transposed(0, 3, 1, 2)
        }

        // Calculate number of image tokens needed after spatial merging
        let numPatchesH = paddedHeight / patchSize
        let numPatchesW = paddedWidth / patchSize
        let mergedPatchesH = numPatchesH / spatialMergeSize
        let mergedPatchesW = numPatchesW / spatialMergeSize
        let numImageTokens = mergedPatchesH * mergedPatchesW

        return PreprocessResult(
            pixels: pixels,
            frames: [THW(1, paddedHeight, paddedWidth)],
            numImageTokens: numImageTokens
        )
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        // Generate structured messages using the message generator
        let messages = Mistral3MessageGenerator().generate(from: input)

        if input.images.isEmpty {
            // No image - just apply chat template
            let promptTokens = try tokenizer.applyChatTemplate(
                messages: messages,
                tools: input.tools,
                additionalContext: input.additionalContext
            )
            let tokensArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
            let mask = ones(like: tokensArray)
            return LMInput(text: .init(tokens: tokensArray, mask: mask), image: nil)
        }

        guard input.images.count == 1 else {
            throw VLMError.singleImageAllowed
        }
        let spatialMergeSize = config.spatialMergeSize ?? 2
        let patchSize = config.imageProcessor.patchSize

        // Apply chat template to get tokenized prompt with image placeholder
        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext
        )

        // Decode to find and replace image placeholder token
        let decoded = tokenizer.decode(tokenIds: promptTokens, skipSpecialTokens: false)

        // Process image to get dimensions
        let preprocessResult = try preprocessImage(
            input.images[0].asCIImage(),
            processing: input.processing,
            patchSize: patchSize,
            spatialMergeSize: spatialMergeSize,
            longestEdge: config.imageProcessor.size.longestEdge
        )

        // Replace the image placeholder token with the correct number of image tokens
        // The chat template should have inserted the imageToken (e.g., "[IMG]") which we need to expand
        if decoded.contains(imageToken) {
            // Split by image token and re-encode with expanded image tokens
            let pieces = decoded.components(separatedBy: imageToken)
            var expandedTokens: [Int] = []

            for (index, piece) in pieces.enumerated() {
                if !piece.isEmpty {
                    let pieceTokens = tokenizer.encode(text: piece)
                    expandedTokens.append(contentsOf: pieceTokens)
                }
                // Add image tokens between pieces (not after the last one)
                if index < pieces.count - 1 {
                    expandedTokens.append(
                        contentsOf: Array(
                            repeating: imageTokenId, count: preprocessResult.numImageTokens))
                }
            }
            promptTokens = expandedTokens
        } else {
            // Fallback: If no image token placeholder found, try to find and replace the single image token ID
            // or insert at the beginning after BOS
            var foundImageToken = false
            var expandedTokens: [Int] = []

            for token in promptTokens {
                if token == imageTokenId && !foundImageToken {
                    // Replace single image token with expanded tokens
                    expandedTokens.append(
                        contentsOf: Array(
                            repeating: imageTokenId, count: preprocessResult.numImageTokens))
                    foundImageToken = true
                } else {
                    expandedTokens.append(token)
                }
            }

            if foundImageToken {
                promptTokens = expandedTokens
            } else {
                // Last resort: insert image tokens after BOS (if present) or at start
                var insertIndex = 0
                if !promptTokens.isEmpty && promptTokens[0] == 1 {
                    insertIndex = 1  // After BOS token
                }
                promptTokens.insert(
                    contentsOf: Array(
                        repeating: imageTokenId, count: preprocessResult.numImageTokens),
                    at: insertIndex
                )
            }
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray)

        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: .init(pixels: preprocessResult.pixels, frames: preprocessResult.frames)
        )
    }
}
