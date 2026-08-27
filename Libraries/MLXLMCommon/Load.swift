// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Load model weights.
///
/// This is typically called via ``GenericModelFactory/load(from:using:configuration:useLatest:progressHandler:)``.
/// This function loads all `safetensor` files in the given `modelDirectory`,
/// calls ``BaseLanguageModel/sanitize(weights:metadata:)`` to allow per-model preprocessing,
/// applies optional quantization, and
/// updates the model with the weights.
public func loadWeights(
    modelDirectory: URL, model: BaseLanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
) throws {
    // load the weights and collect metadata from the first safetensor file
    var weights = [String: MLXArray]()
    var metadata = [String: String]()
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    var safetensorURLs = [URL]()
    var mlxstURLs = [URL]()
    for case let url as URL in enumerator {
        switch url.pathExtension {
        case "safetensors": safetensorURLs.append(url)
        case "mlxst": mlxstURLs.append(url)
        default: break
        }
    }

    // MLX_MMAP_WEIGHTS=1: prefer page-aligned `.mlxst` repacks, loaded as
    // file-backed zero-copy views (see MmapWeights.swift). Each successful
    // mmap load replaces its sibling `.safetensors` shard; on any failure the
    // shard falls through to the stock loader below.
    var mmapLoadedSiblings = Set<String>()
    if ProcessInfo.processInfo.environment["MLX_MMAP_WEIGHTS"] == "1" {
        for url in mlxstURLs {
            guard let (w, m) = try mmapLoadArraysAndMetadata(url: url) else { continue }
            // A bf16 tensor served as a read-only mapped view must never meet
            // the MLX_VLM_DTYPE=float16 cast below: the cast's input becomes
            // sole-owner at eval, the core donates, and the copy kernel writes
            // f16 bits into the PROT_READ mapping — garbage weights (see the
            // donation note on mmapLoadArraysAndMetadata). A repack made with
            // convertingBF16To: .float16 carries no bf16 tensors; a stale
            // unconverted one falls back to the stock loader, which casts
            // freshly-allocated arrays safely.
            if ProcessInfo.processInfo.environment["MLX_VLM_DTYPE"] == "float16",
                w.values.contains(where: { $0.dtype == .bfloat16 })
            {
                print(
                    "[mmap] \(url.lastPathComponent): bf16 tensors + MLX_VLM_DTYPE=float16 — "
                        + "using safetensors loader; repack with convertingBF16To: .float16")
                continue
            }
            for (key, value) in w {
                weights[key] = value
            }
            if metadata.isEmpty {
                metadata = m
            }
            mmapLoadedSiblings.insert(
                url.deletingPathExtension().appendingPathExtension("safetensors").path)
        }
    }

    for url in safetensorURLs where !mmapLoadedSiblings.contains(url.path) {
        let (w, m) = try loadArraysAndMetadata(url: url)
        for (key, value) in w {
            weights[key] = value
        }
        if metadata.isEmpty {
            metadata = m
        }
    }

    // per-model cleanup (models can inspect metadata to customize behavior)
    weights = model.sanitize(weights: weights, metadata: metadata)

    // MLX_VLM_DTYPE=float16: cast bf16 weights to fp16 at load. M1/M2 GPUs
    // have no native bf16 — MLX emulates it, costing ~20-30% on prefill.
    if ProcessInfo.processInfo.environment["MLX_VLM_DTYPE"] == "float16" {
        weights = weights.mapValues { $0.dtype == .bfloat16 ? $0.asType(.float16) : $0 }
    }

    // quantize if needed
    if quantization != nil || perLayerQuantization != nil {
        quantize(model: model) { path, module in
            if weights["\(path).scales"] != nil {
                if let perLayerQuantization {
                    return perLayerQuantization.quantization(layer: path)?.asTuple
                } else {
                    return quantization?.asTuple
                }
            } else {
                return nil
            }
        }
    }

    // apply the loaded weights
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.all])

    eval(model)
}
