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
        MmapWeights.register(self)
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
///
/// WARNING — downstream dtype casts of the returned views are UNSUPPORTED.
/// Field finding (2026-08): `MLX_MMAP_WEIGHTS=1` + `MLX_VLM_DTYPE=float16`
/// broke generation (prefill ran, then immediate EOS, empty output). Root
/// cause, traced through the vendored core: `loadWeights`'s
/// `mapValues { $0.asType(.float16) }` drops the original view wrapper, so at
/// eval time the cast's input has `use_count == 1` on both desc and data —
/// `array::is_donatable()` (array.h:294) is true. `AsType::eval_gpu` →
/// `copy_gpu(CopyType::Vector)` → `set_copy_output_data`
/// (backend/common/copy.h:30): same itemsize (bf16 == f16 == 2 B) →
/// `is_donatable(in, out)` (backend/common/utils.h:185) → `out.copy_shared_buffer(in)`
/// — the copy kernel then writes f16 bits IN-PLACE into this `PROT_READ`,
/// `MAP_SHARED` file-backed buffer. GPU writes into a read-only mapping are
/// dropped/undefined, so every bf16 tensor (norms, scales, embeddings) keeps
/// its bf16 bit patterns reinterpreted as f16 → garbage weights → instant EOS.
/// The PORT-PLAN donation analysis only covered inference (module-held weights
/// have use_count > 1); this load-time cast window was the gap. The supported
/// path for an fp16 model is repack-time conversion:
/// `MLXSTFile.repack(safetensors:to:convertingBF16To: .float16)`.
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

/// Process-wide view of live weight mappings, so a host can ask the kernel to
/// bring a model's pages back before it is needed.
///
/// Mapped weights are clean file-backed pages, which macOS evicts first under
/// pressure. A model that idles for minutes between batches comes back with
/// almost nothing resident (measured 704 KB of 2.3 GB) and then faults the
/// whole file in 16 KB pages, in the random order decode touches them, while
/// generating. `MADV_WILLNEED` turns that into one sequential readahead the
/// moment a batch is known to be coming; it is a no-op for resident pages.
public enum MmapWeights {
    private struct Weak { weak var holder: WeightMapHolder? }
    nonisolated(unsafe) private static var holders: [Weak] = []
    private static let lock = NSLock()

    static func register(_ holder: WeightMapHolder) {
        lock.lock(); defer { lock.unlock() }
        holders.removeAll { $0.holder == nil }
        holders.append(Weak(holder: holder))
    }

    /// Bytes currently mapped across all live weight files.
    public static var mappedBytes: Int {
        lock.lock(); defer { lock.unlock() }
        return holders.compactMap { $0.holder?.length }.reduce(0, +)
    }

    /// Ask the kernel to read every live mapping back into memory
    /// (asynchronous readahead). Returns the number of bytes advised.
    @discardableResult
    public static func prefetchAll() -> Int {
        lock.lock(); defer { lock.unlock() }
        var advised = 0
        for w in holders {
            guard let h = w.holder else { continue }
            if madvise(h.base, h.length, MADV_WILLNEED) == 0 { advised += h.length }
        }
        return advised
    }
}
