// Copyright © 2026 Apple Inc.

import Cmlx
import Darwin
import Foundation
import MLX

/// Keeps an mmap'd weight file alive until the last tensor view over it is
/// released: every `MLXArray` created by ``mmapLoadArraysAndMetadata(url:)``
/// captures the holder in its finalizer, so `munmap` runs after the final
/// view's death (model unload → container released → views die → munmap).
final class WeightMapHolder {
    let base: UnsafeMutableRawPointer
    let length: Int

    init(base: UnsafeMutableRawPointer, length: Int) {
        self.base = base
        self.length = length
    }

    deinit {
        munmap(base, length)
    }
}

/// Diagnostics from the most recent ``mmapLoadArraysAndMetadata(url:)`` call.
/// Test / telemetry hook only.
enum MmapWeightsDiagnostics {
    /// tensor counts from the last load: `mapped` passed the no-copy guard,
    /// `copied` did not (the core silently memcpy'd into anonymous memory)
    nonisolated(unsafe) static var lastLoad: (mapped: Int, copied: Int) = (0, 0)
}

/// Load a page-aligned `.mlxst` file (see ``MLXSTFile``) as file-backed
/// zero-copy views: the file is mmap'd (`PROT_READ`, `MAP_SHARED`) and each
/// tensor becomes an `MLXArray(rawPointer:)` view into the mapping, which MLX
/// hands to Metal via `newBufferWithBytesNoCopy`. The bytes never appear as
/// dirty/anonymous memory in the process footprint and the OS can evict and
/// refault them under pressure.
///
/// Returns nil when the file is not `.mlxst`, or when the no-copy guard below
/// detects that the core silently copied instead of mapping — in both cases
/// the caller falls back to the stock safetensors loader.
func mmapLoadArraysAndMetadata(url: URL) throws -> ([String: MLXArray], [String: String])? {
    guard let header = try MLXSTFile.readHeader(url: url) else { return nil }

    let fd = open(url.path, O_RDONLY)
    guard fd >= 0 else {
        throw MLXSTError.invalidFile("open(\(url.path)) failed: \(String(cString: strerror(errno)))")
    }
    var status = stat()
    guard fstat(fd, &status) == 0 else {
        let error = String(cString: strerror(errno))
        close(fd)
        throw MLXSTError.invalidFile("fstat(\(url.path)) failed: \(error)")
    }
    let length = Int(status.st_size)
    if let last = header.tensors.map({ $0.offset + $0.nbytes }).max(),
        header.payloadOffset + last > length
    {
        close(fd)
        throw MLXSTError.invalidFile("payload extends past EOF (\(length) bytes)")
    }
    guard let base = mmap(nil, length, PROT_READ, MAP_SHARED, fd, 0), base != MAP_FAILED else {
        let error = String(cString: strerror(errno))
        close(fd)
        throw MLXSTError.invalidFile("mmap(\(url.path)) failed: \(error)")
    }
    close(fd)
    let holder = WeightMapHolder(base: base, length: length)

    var weights = [String: MLXArray]()
    var copied = 0
    for tensor in header.tensors {
        guard let dtype = MLXSTFile.mlxDTypes[tensor.dtype] else {
            throw MLXSTError.unsupportedDType(key: tensor.key, dtype: tensor.dtype)
        }
        let pointer = base.advanced(by: header.payloadOffset + tensor.offset)
        let array = MLXArray(
            rawPointer: pointer, tensor.shape, dtype: dtype,
            finalizer: { [holder] in _ = holder })

        // No-copy guard: if `allocator::make_buffer` (newBufferWithBytesNoCopy)
        // rejects the pointer, the core silently memcpy's into a fresh
        // anonymous buffer — correct but it defeats the whole point. Detect it
        // by comparing the array's backing pointer with the mapped bytes.
        if tensor.nbytes > 0,
            UnsafeRawPointer(mlx_array_data_uint8(array.ctx)) != UnsafeRawPointer(pointer)
        {
            copied += 1
        }
        weights[tensor.key] = array
    }
    MmapWeightsDiagnostics.lastLoad = (header.tensors.count - copied, copied)

    if copied > 0 {
        print(
            "[mmap] \(url.lastPathComponent): \(copied)/\(header.tensors.count) tensors were "
                + "copied instead of mapped; falling back to safetensors load")
        return nil
    }
    return (weights, header.metadata)
}
