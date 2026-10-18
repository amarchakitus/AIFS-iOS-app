import CryptoKit
import Foundation
import Testing
@testable import AIFS_iOS_app

/// The model download, against a bundle served from file:// URLs (no network): whole
/// downloads land intact, corrupt files are refused, and a stopped download resumes.
struct ModelDownloadTests {
    private let root: URL
    private var source: URL { root.appending(path: "source") }
    private var staging: URL { root.appending(path: "staging") }
    private var bundle: URL { root.appending(path: "bundle") }

    init() throws {
        root = FileManager.default.temporaryDirectory.appending(path: "model-download-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: source.appending(path: "coreml"), withIntermediateDirectories: true)
        try Data((0 ..< 50_000).map { UInt8($0 % 251) }).write(to: source.appending(path: "weights.safetensors"))
        try Data("{\"format\":\"test\"}".utf8).write(to: source.appending(path: "manifest.json"))
        try Data(repeating: 7, count: 3_000).write(to: source.appending(path: "coreml/proc_00.bin"))
    }

    private let paths = ["weights.safetensors", "manifest.json", "coreml/proc_00.bin"]

    private func writeIndex(corrupting bad: String? = nil) throws {
        let files = try paths.map { path -> [String: Any] in
            let data = try Data(contentsOf: source.appending(path: path))
            let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            return ["path": path, "size": data.count, "sha256": path == bad ? String(repeating: "0", count: 64) : hash]
        }
        try JSONSerialization.data(withJSONObject: ["files": files]).write(to: source.appending(path: "files.json"))
    }

    private func download() async throws {
        let source = source
        try await ModelDownloader.download(
            from: { source.appending(path: $0) }, staging: staging, into: bundle, progress: { _, _ in })
    }

    @Test func downloadsEveryFileIntact() async throws {
        try writeIndex()
        try await download()
        for path in paths {
            #expect(try Data(contentsOf: bundle.appending(path: path)) == Data(contentsOf: source.appending(path: path)))
        }
        #expect(!FileManager.default.fileExists(atPath: staging.path))  // renamed into place
        try? FileManager.default.removeItem(at: root)
    }

    @Test func refusesAFileThatFailsItsChecksum() async throws {
        try writeIndex(corrupting: "manifest.json")
        await #expect(throws: (any Error).self) { try await download() }
        // Nothing half-made in place, and the bad file not kept even as a partial.
        #expect(!FileManager.default.fileExists(atPath: bundle.path))
        #expect(!FileManager.default.fileExists(atPath: staging.appending(path: "manifest.json").path))
        #expect(!FileManager.default.fileExists(atPath: staging.appending(path: "manifest.json.download").path))
        try? FileManager.default.removeItem(at: root)
    }

    @Test func resumesWithoutRefetchingVerifiedFiles() async throws {
        try writeIndex()
        // A previous attempt already verified the weights; then the source loses them. If
        // the download fetched them again it would fail.
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        try FileManager.default.copyItem(
            at: source.appending(path: "weights.safetensors"), to: staging.appending(path: "weights.safetensors"))
        try FileManager.default.removeItem(at: source.appending(path: "weights.safetensors"))
        try await download()
        #expect(FileManager.default.fileExists(atPath: bundle.appending(path: "weights.safetensors").path))
        #expect(FileManager.default.fileExists(atPath: bundle.appending(path: "coreml/proc_00.bin").path))
        try? FileManager.default.removeItem(at: root)
    }
}
