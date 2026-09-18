import Foundation
import Testing

@testable import MLXVLM

/// Boundary logic for the Qwen3.5 prompt-prefix KV cache (`Qwen35.recordBoundary`),
/// the same guard Mistral3 carries: a boundary equal to the token count leaves an
/// empty suffix slice, and prefilling a zero-length sequence crashes MLX's reshape
/// and takes the helper down with it (Sentry RECALL-ADHD-3A on the Mistral3 path).
/// Pure function, so the guard is testable without loading model weights.
struct Qwen35PrefixCacheTests {

    private func tokens(_ count: Int) -> [Int32] { (0 ..< count).map { Int32($0) } }

    @Test("An identical text-only request leaves no suffix → skip")
    func identicalTextOnlyRequestSkips() {
        let flat = tokens(300)
        #expect(Qwen35.recordBoundary(flat: flat, lastTokens: flat, firstImage: flat.count) == nil)
    }

    @Test("A request that is a strict prefix of the previous one → skip")
    func currentIsPrefixOfPreviousSkips() {
        let prev = tokens(300)
        let flat = Array(prev[0 ..< 200])
        #expect(Qwen35.recordBoundary(flat: flat, lastTokens: prev, firstImage: flat.count) == nil)
    }

    @Test("Shared prefix then divergence → records at the longest common prefix")
    func divergingRecordsAtLCP() {
        var prev = tokens(200); prev += (0 ..< 100).map { Int32(1000 + $0) }
        var flat = tokens(200); flat += (0 ..< 100).map { Int32(2000 + $0) }
        #expect(Qwen35.recordBoundary(flat: flat, lastTokens: prev, firstImage: flat.count) == 200)
    }

    @Test("The boundary never reaches past the first image token")
    func boundaryCappedAtFirstImage() {
        // Image features are per-request: their KV must never enter the snapshot.
        let flat = tokens(400)
        #expect(Qwen35.recordBoundary(flat: flat, lastTokens: flat, firstImage: 150) == 150)
    }

    @Test("A prefix shorter than the minimum is not worth snapshotting → skip")
    func shortPrefixSkips() {
        var prev = tokens(50); prev += (0 ..< 100).map { Int32(1000 + $0) }
        var flat = tokens(50); flat += (0 ..< 100).map { Int32(2000 + $0) }
        #expect(Qwen35.recordBoundary(flat: flat, lastTokens: prev, firstImage: flat.count) == nil)
    }
}
