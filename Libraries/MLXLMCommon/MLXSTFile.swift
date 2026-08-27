// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Errors thrown by ``MLXSTFile`` and the mmap weight loader.
public enum MLXSTError: LocalizedError {
    case invalidFile(String)
    case unsupportedDType(key: String, dtype: String)

    public var errorDescription: String? {
        switch self {
        case .invalidFile(let message): return "invalid mlxst file: \(message)"
        case .unsupportedDType(let key, let dtype):
            return "unsupported dtype '\(dtype)' for tensor '\(key)'"
        }
    }
}

/// The `.mlxst` repacked-weights container.
///
/// A lossless repack of a `.safetensors` shard with the same JSON header schema,
/// but with the payload base and every tensor's start offset aligned to
/// ``pageSize``. That alignment is what lets the file be mmap'd and each tensor
/// served to Metal via `newBufferWithBytesNoCopy` (see `MmapWeights.swift`)
/// instead of being read and copied into anonymous memory.
///
/// Wire format:
/// ```
/// 8 B   magic  "MLXST\0\1\0"
/// 8 B   header_len (LE u64)
///       JSON header — safetensors schema (dtype/shape/data_offsets per tensor
///       + __metadata__), data_offsets relative to the payload base
///       zero pad so the payload base ≡ 0 (mod pageSize)
///       each tensor zero-padded to the next pageSize boundary
/// ```
///
/// The reader computes the payload base by rounding `16 + header_len` up to
/// `pageSize`, so the writer MUST always pad the header to that boundary.
public enum MLXSTFile {

    /// Alignment of the payload base and of each tensor's start offset
    /// (`vm_page_size` on Apple Silicon).
    public static let pageSize = 16384

    /// 8-byte magic: "MLXST" NUL, version 1.0
    public static let magic = Data([0x4D, 0x4C, 0x58, 0x53, 0x54, 0x00, 0x01, 0x00])

    /// bytes per element for each safetensors dtype the format can carry
    static let dtypeSizes: [String: Int] = [
        "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E8M0": 1,
        "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
        "U32": 4, "I32": 4, "F32": 4,
        "U64": 8, "I64": 8, "F64": 8,
        "C64": 8,
    ]

    /// safetensors dtype string → MLX dtype, for the dtypes MLX can represent
    static let mlxDTypes: [String: DType] = [
        "BOOL": .bool, "U8": .uint8, "I8": .int8,
        "U16": .uint16, "I16": .int16, "F16": .float16, "BF16": .bfloat16,
        "U32": .uint32, "I32": .int32, "F32": .float32,
        "U64": .uint64, "I64": .int64, "F64": .float64,
        "C64": .complex64,
    ]

    struct Tensor {
        let key: String
        let dtype: String
        let shape: [Int]
        /// byte offset relative to the payload base
        let offset: Int
        let nbytes: Int
    }

    struct Header {
        let tensors: [Tensor]
        let metadata: [String: String]
        /// absolute file offset of the payload base (multiple of ``pageSize``)
        let payloadOffset: Int
    }

    // MARK: - Reading

    /// True if the file at `url` starts with the `.mlxst` magic.
    public static func isMLXST(url: URL) -> Bool {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return false }
        defer { try? handle.close() }
        return (try? handle.read(upToCount: magic.count)) == magic
    }

    /// Parse the header of an `.mlxst` file. Returns nil if the magic does not
    /// match (not an `.mlxst` file); throws if the file is malformed.
    static func readHeader(url: URL) throws -> Header? {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        guard try handle.read(upToCount: magic.count) == magic else { return nil }
        guard let lengthData = try handle.read(upToCount: 8), lengthData.count == 8 else {
            throw MLXSTError.invalidFile("truncated header length")
        }
        let headerLength = lengthData.withUnsafeBytes { Int(UInt64(littleEndian: $0.load(as: UInt64.self))) }
        guard headerLength > 0, headerLength < 100_000_000 else {
            throw MLXSTError.invalidFile("implausible header length \(headerLength)")
        }
        guard let headerData = try handle.read(upToCount: headerLength),
            headerData.count == headerLength
        else {
            throw MLXSTError.invalidFile("truncated header")
        }

        let (entries, metadata) = try parseHeaderJSON(headerData)
        let tensors = try entries.map { key, entry -> Tensor in
            guard entry.offsets.1 - entry.offsets.0 == entry.nbytes else {
                throw MLXSTError.invalidFile(
                    "tensor '\(key)': data_offsets span \(entry.offsets.1 - entry.offsets.0), expected \(entry.nbytes)"
                )
            }
            return Tensor(
                key: key, dtype: entry.dtype, shape: entry.shape,
                offset: entry.offsets.0, nbytes: entry.nbytes)
        }

        let pageSize = metadata["mlxst_page_size"].flatMap { Int($0) } ?? Self.pageSize
        let payloadOffset = roundUp(magic.count + 8 + headerLength, to: pageSize)
        return Header(tensors: tensors, metadata: metadata, payloadOffset: payloadOffset)
    }

    // MARK: - Writing (repack)

    /// Repack a `.safetensors` shard into a page-aligned `.mlxst` file.
    ///
    /// Lossless by default: same JSON header schema, same tensor bytes.
    /// Tensors are laid out in source-offset order, each padded to the next
    /// ``pageSize`` boundary. Writes to `<destination>.tmp` in the same
    /// directory and atomically renames on success, so an interrupted repack
    /// never leaves a half-written `.mlxst` behind.
    ///
    /// `convertingBF16To: .float16` additionally converts every BF16 tensor to
    /// Float16 during the streamed write (round-to-nearest-even via Float32,
    /// identical to an MLX `asType` cast). This is the supported way to get an
    /// fp16 model out of a bf16 checkpoint under the mmap loader — casting the
    /// mmap-backed views at load time is NOT safe (see the donation note on
    /// `mmapLoadArraysAndMetadata`). The conversion is recorded in the header
    /// metadata; use ``convertedDtype(url:)`` to detect a stale repack when
    /// the desired dtype changes.
    public static func repack(
        safetensors source: URL, to destination: URL, convertingBF16To targetDType: DType? = nil
    ) throws {
        switch targetDType {
        case .none, .float16: break
        default:
            throw MLXSTError.unsupportedDType(
                key: "convertingBF16To", dtype: String(describing: targetDType!))
        }
        let sourceHandle = try FileHandle(forReadingFrom: source)
        defer { try? sourceHandle.close() }

        // vanilla safetensors: 8 B header_len (LE u64) + JSON header + payload
        guard let lengthData = try sourceHandle.read(upToCount: 8), lengthData.count == 8 else {
            throw MLXSTError.invalidFile("truncated safetensors header length")
        }
        let headerLength = lengthData.withUnsafeBytes { Int(UInt64(littleEndian: $0.load(as: UInt64.self))) }
        guard let headerData = try sourceHandle.read(upToCount: headerLength),
            headerData.count == headerLength
        else {
            throw MLXSTError.invalidFile("truncated safetensors header")
        }
        let sourcePayloadOffset = 8 + headerLength
        let (entries, sourceMetadata) = try parseHeaderJSON(headerData)

        // destination layout: source-offset order, each tensor start page-aligned
        var newHeader = [String: Any]()
        var layout = [(key: String, sourceOffset: Int, nbytes: Int, convert: Bool)]()
        var cursor = 0
        for (key, entry) in entries.sorted(by: { $0.entry.offsets.0 < $1.entry.offsets.0 }) {
            guard entry.offsets.1 - entry.offsets.0 == entry.nbytes else {
                throw MLXSTError.invalidFile(
                    "tensor '\(key)': data_offsets span \(entry.offsets.1 - entry.offsets.0), expected \(entry.nbytes)"
                )
            }
            // BF16 → F16 keeps element count and item size (2 bytes), so the
            // destination byte span is unchanged; only the dtype tag differs
            let convert = targetDType == .float16 && entry.dtype == "BF16"
            newHeader[key] = [
                "dtype": convert ? "F16" : entry.dtype,
                "shape": entry.shape,
                "data_offsets": [cursor, cursor + entry.nbytes],
            ]
            layout.append((key, entry.offsets.0, entry.nbytes, convert))
            cursor = roundUp(cursor + entry.nbytes, to: pageSize)
        }

        // Source metadata is carried through untouched — models inspect it in
        // sanitize() (e.g. `format == "mlx"` gates transposes in Qwen35), so
        // the repack marker uses `mlxst_`-prefixed keys instead of overriding.
        var metadata = sourceMetadata
        metadata["mlxst_version"] = "0"
        metadata["mlxst_page_size"] = String(pageSize)
        if targetDType == .float16 {
            // recorded whenever the conversion was REQUESTED (even for a shard
            // with no BF16 tensors) so a caller can compare against the
            // currently-desired dtype without re-parsing tensor entries
            metadata["mlxst_converted"] = "bfloat16->float16"
        }
        newHeader["__metadata__"] = metadata

        let headerBytes = try JSONSerialization.data(
            withJSONObject: newHeader, options: [.sortedKeys])
        let prePayload = magic.count + 8 + headerBytes.count
        let payloadOffset = roundUp(prePayload, to: pageSize)

        let temporary = destination.appendingPathExtension("tmp")
        FileManager.default.createFile(atPath: temporary.path, contents: nil)
        do {
            let destinationHandle = try FileHandle(forWritingTo: temporary)
            defer { try? destinationHandle.close() }

            try destinationHandle.write(contentsOf: magic)
            var headerLengthLE = UInt64(headerBytes.count).littleEndian
            try destinationHandle.write(contentsOf: Data(bytes: &headerLengthLE, count: 8))
            try destinationHandle.write(contentsOf: headerBytes)
            if payloadOffset > prePayload {
                try destinationHandle.write(contentsOf: Data(count: payloadOffset - prePayload))
            }

            let chunkSize = 1 << 20
            for (key, sourceOffset, nbytes, convert) in layout {
                try sourceHandle.seek(toOffset: UInt64(sourcePayloadOffset + sourceOffset))
                var remaining = nbytes
                while remaining > 0 {
                    guard let chunk = try sourceHandle.read(upToCount: min(chunkSize, remaining)),
                        !chunk.isEmpty, !(convert && chunk.count % 2 != 0)
                    else {
                        throw MLXSTError.invalidFile("unexpected EOF copying tensor '\(key)'")
                    }
                    try destinationHandle.write(contentsOf: convert ? convertBF16ToF16(chunk) : chunk)
                    remaining -= chunk.count
                }
                let padding = roundUp(nbytes, to: pageSize) - nbytes
                if padding > 0 {
                    try destinationHandle.write(contentsOf: Data(count: padding))
                }
            }
        } catch {
            try? FileManager.default.removeItem(at: temporary)
            throw error
        }

        // atomic replace (rename(2)); never leaves a partial .mlxst at `destination`
        guard rename(temporary.path, destination.path) == 0 else {
            let error = String(cString: strerror(errno))
            try? FileManager.default.removeItem(at: temporary)
            throw MLXSTError.invalidFile("rename to \(destination.path) failed: \(error)")
        }
    }

    /// The dtype conversion recorded in an `.mlxst` file's metadata, or nil
    /// for a lossless repack. Callers compare this against the dtype they
    /// currently want (e.g. `.float16` when `MLX_VLM_DTYPE=float16`) and
    /// re-repack when it differs. Throws if the file is not `.mlxst`.
    public static func convertedDtype(url: URL) throws -> DType? {
        guard let header = try readHeader(url: url) else {
            throw MLXSTError.invalidFile("\(url.lastPathComponent) is not an .mlxst file")
        }
        switch header.metadata["mlxst_converted"] {
        case "bfloat16->float16": return .float16
        default: return nil
        }
    }

    /// bf16 → f16, element-wise: widen to Float32 (bf16 bits are the top half
    /// of the f32 pattern) then narrow to Float16 with the hardware's
    /// round-to-nearest-even — the same result an MLX `asType` cast produces.
    private static func convertBF16ToF16(_ chunk: Data) -> Data {
        var out = Data(count: chunk.count)
        out.withUnsafeMutableBytes { destination in
            chunk.withUnsafeBytes { source in
                let src = source.bindMemory(to: UInt16.self)
                let dst = destination.bindMemory(to: UInt16.self)
                for i in 0 ..< src.count {
                    let value = Float(bitPattern: UInt32(src[i]) << 16)
                    dst[i] = Float16(value).bitPattern
                }
            }
        }
        return out
    }

    // MARK: - Shared helpers

    private struct HeaderEntry {
        let dtype: String
        let shape: [Int]
        let offsets: (Int, Int)
        let nbytes: Int
    }

    /// Parse a safetensors-schema JSON header into tensor entries + string metadata.
    private static func parseHeaderJSON(_ data: Data) throws -> (
        entries: [(key: String, entry: HeaderEntry)], metadata: [String: String]
    ) {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw MLXSTError.invalidFile("header is not a JSON object")
        }

        var metadata = [String: String]()
        if let raw = json["__metadata__"] as? [String: Any] {
            for (key, value) in raw {
                if let string = value as? String { metadata[key] = string }
            }
        }

        var entries = [(key: String, entry: HeaderEntry)]()
        for (key, value) in json where key != "__metadata__" {
            guard let entry = value as? [String: Any],
                let dtype = entry["dtype"] as? String,
                let shape = entry["shape"] as? [Int],
                let offsets = entry["data_offsets"] as? [Int], offsets.count == 2
            else {
                throw MLXSTError.invalidFile("malformed entry for tensor '\(key)'")
            }
            guard let itemSize = dtypeSizes[dtype] else {
                throw MLXSTError.unsupportedDType(key: key, dtype: dtype)
            }
            let nbytes = shape.reduce(itemSize, *)
            entries.append(
                (key, HeaderEntry(dtype: dtype, shape: shape, offsets: (offsets[0], offsets[1]), nbytes: nbytes))
            )
        }
        return (entries, metadata)
    }

    static func roundUp(_ value: Int, to alignment: Int) -> Int {
        (value + alignment - 1) / alignment * alignment
    }
}
