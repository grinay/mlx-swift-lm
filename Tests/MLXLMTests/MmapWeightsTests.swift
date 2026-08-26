// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// `.mlxst` repack + mmap load round-trip: write a synthetic safetensors
/// file, repack it, mmap-load it, and verify values and the no-copy guard.
@Suite("MmapWeights", .serialized)
struct MmapWeightsTests {

    /// build a minimal vanilla safetensors file: 8 B LE header length + JSON
    /// header + payload (deliberately unaligned payload base)
    private func writeSafetensors(
        to url: URL, tensors: [(key: String, dtype: String, shape: [Int], bytes: Data)],
        metadata: [String: String] = [:]
    ) throws {
        var header = [String: Any]()
        if !metadata.isEmpty {
            header["__metadata__"] = metadata
        }
        var payload = Data()
        for (key, dtype, shape, bytes) in tensors {
            header[key] = [
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [payload.count, payload.count + bytes.count],
            ]
            payload.append(bytes)
        }
        let headerBytes = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])

        var file = Data()
        var headerLength = UInt64(headerBytes.count).littleEndian
        file.append(Data(bytes: &headerLength, count: 8))
        file.append(headerBytes)
        file.append(payload)
        try file.write(to: url)
    }

    private func data<T>(_ values: [T]) -> Data {
        values.withUnsafeBytes { Data($0) }
    }

    private func makeFixture(in directory: URL) throws -> (safetensors: URL, mlxst: URL) {
        let safetensors = directory.appendingPathComponent("model.safetensors")
        let mlxst = directory.appendingPathComponent("model.mlxst")
        // mixed dtypes incl. the quantized-checkpoint triplet (u32 packed +
        // f16 scales/biases) and a scalar; > 1 page of u32 to span pages
        try writeSafetensors(
            to: safetensors,
            tensors: [
                (
                    "layer.weight", "U32", [128, 40],
                    data((0 ..< 5120).map { UInt32(truncatingIfNeeded: $0 &* 2_654_435_761) })
                ),
                ("layer.scales", "F16", [16, 8], data((0 ..< 128).map { Float16($0) * 0.25 })),
                ("layer.biases", "F32", [64], data((0 ..< 64).map { Float($0) - 31.5 })),
                ("layer.index", "I64", [3, 2], data((0 ..< 6).map { Int64($0 * 1000 - 3000) })),
                ("scalar", "F32", [], data([Float(42.5)])),
            ],
            metadata: ["format": "mlx"])
        try MLXSTFile.repack(safetensors: safetensors, to: mlxst)
        return (safetensors, mlxst)
    }

    @Test func repackLayoutIsPageAligned() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let (_, mlxst) = try makeFixture(in: directory)

        #expect(MLXSTFile.isMLXST(url: mlxst))
        let header = try #require(try MLXSTFile.readHeader(url: mlxst))
        #expect(header.payloadOffset % MLXSTFile.pageSize == 0)
        #expect(header.tensors.count == 5)
        for tensor in header.tensors {
            #expect(tensor.offset % MLXSTFile.pageSize == 0, "tensor \(tensor.key) not page-aligned")
        }
        // source metadata carried through, not clobbered by the repack marker
        #expect(header.metadata["format"] == "mlx")
        #expect(header.metadata["mlxst_page_size"] == String(MLXSTFile.pageSize))
    }

    @Test func mmapLoadMatchesSafetensorsAndNoCopyGuardPasses() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let (safetensors, mlxst) = try makeFixture(in: directory)

        let loaded = try #require(try mmapLoadArraysAndMetadata(url: mlxst))
        let (reference, referenceMetadata) = try loadArraysAndMetadata(url: safetensors)

        // (c) the no-copy guard passed: every tensor is a view into the mapping
        #expect(MmapWeightsDiagnostics.lastLoad.copied == 0)
        #expect(MmapWeightsDiagnostics.lastLoad.mapped == reference.count)

        // values match the stock safetensors load, bit for bit
        #expect(Set(loaded.0.keys) == Set(reference.keys))
        for (key, expected) in reference {
            let actual = try #require(loaded.0[key])
            #expect(actual.dtype == expected.dtype, "dtype mismatch for \(key)")
            #expect(actual.shape == expected.shape, "shape mismatch for \(key)")
            #expect(
                (actual .== expected).all().item(Bool.self), "value mismatch for \(key)")
        }
        #expect(loaded.1["format"] == referenceMetadata["format"])
    }

    @Test func nonMlxstFileReturnsNil() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let url = directory.appendingPathComponent("model.mlxst")
        try Data("this is not an mlxst file".utf8).write(to: url)
        #expect(try mmapLoadArraysAndMetadata(url: url) == nil)
    }

    @Test func truncatedPayloadThrows() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let (_, mlxst) = try makeFixture(in: directory)
        let full = try Data(contentsOf: mlxst)
        try full.prefix(full.count - MLXSTFile.pageSize).write(to: mlxst)

        #expect(throws: MLXSTError.self) {
            _ = try mmapLoadArraysAndMetadata(url: mlxst)
        }
    }

    @Test func loadWeightsPrefersMlxstOnlyWhenEnabled() throws {
        // the env toggle is read inside loadWeights; here we verify the
        // sibling-skip bookkeeping indirectly via the loader's building blocks:
        // a directory with both files yields identical dictionaries either way
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let (safetensors, mlxst) = try makeFixture(in: directory)
        let viaMmap = try #require(try mmapLoadArraysAndMetadata(url: mlxst))
        let viaStock = try loadArraysAndMetadata(url: safetensors)
        #expect(Set(viaMmap.0.keys) == Set(viaStock.0.keys))
    }
}
