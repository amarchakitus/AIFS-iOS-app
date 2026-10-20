import CryptoKit
import Foundation

/// Where each model's bundle is published, as `local/publish_model.py` uploads it: for
/// AIFS, the converted weights, graph, Core ML processor and regridding operators; for
/// Aurora and GraphCast, their Core ML programs, tables and metadata. Pinned to one commit
/// each, so a later upload can never hand an installed app a bundle it doesn't understand.
nonisolated enum ModelSource {
    static func repo(for model: ForecastModel) -> (repo: String, revision: String) {
        switch model {
        case .aifs: ("amarchakitus/aifs-single-v2-mlx", "ed19b8327b69a1027e6b5a26a6fad52f05d0e569")
        case .aurora: ("amarchakitus/aurora-1.5-coreml", "4cbc8ae79c074bf16f7ab2f57f74524e4f1a8dfa")
        case .graphcast: ("amarchakitus/graphcast-operational-coreml", "ba7d5ec47c046aa0f9f56eb3521ded7a84354298")
        }
    }

    /// Whether `model` can be downloaded (a commit is pinned).
    static func isPublished(_ model: ForecastModel) -> Bool { !repo(for: model).revision.isEmpty }

    static func url(_ model: ForecastModel, _ path: String) -> URL {
        let (repo, revision) = repo(for: model)
        return URL(string: "https://huggingface.co/\(repo)/resolve/\(revision)/")!.appending(path: path)
    }
}

/// `files.json` at the root of the repo: every file of the bundle with its size and hash.
nonisolated struct BundleIndex: Codable, Sendable {
    struct File: Codable, Sendable {
        let path: String
        let size: Int64
        let sha256: String
    }
    let files: [File]
    var totalBytes: Int64 { files.reduce(0) { $0 + $1.size } }
}

/// Downloads the bundle into a staging folder beside the real one, verifying every file,
/// then renames it into place: a half-finished download is never mistaken for a model,
/// and a stopped one resumes, skipping the files it already has.
nonisolated enum ModelDownloader {
    static func staging(for model: ForecastModel) -> URL {
        AppPaths.documents.appending(path: AppPaths.bundle(for: model).lastPathComponent + ".partial", directoryHint: .isDirectory)
    }

    /// `model`'s published bundle into the app's Documents. `progress(bytesDone, bytesTotal)`.
    static func download(_ model: ForecastModel, progress: @escaping @Sendable (Int64, Int64) -> Void) async throws {
        try await download(from: { ModelSource.url(model, $0) }, staging: staging(for: model),
                           into: AppPaths.bundle(for: model), progress: progress)
    }

    /// The bundle whose files `source(path)` locates, staged in `staging` and moved to
    /// `bundle` once every file is verified. (Separate from the app's paths for tests,
    /// which serve a bundle from `file://` URLs.)
    static func download(
        from source: @escaping @Sendable (String) -> URL, staging: URL, into bundle: URL,
        progress: @escaping @Sendable (Int64, Int64) -> Void
    ) async throws {
        let (data, response) = try await URLSession.shared.data(from: source("files.json"))
        try check(response)
        let index = try JSONDecoder().decode(BundleIndex.self, from: data)
        let total = index.totalBytes
        let fm = FileManager.default

        var have: Int64 = 0
        var todo: [BundleIndex.File] = []
        for file in index.files {
            let size = (try? fm.attributesOfItem(atPath: staging.appending(path: file.path).path)[.size] as? Int64) ?? nil
            // Only verified files are ever moved to their final name in staging.
            if size == file.size { have += file.size } else { todo.append(file) }
        }
        let free = (try? AppPaths.documents.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
            .volumeAvailableCapacityForImportantUsage ?? .max
        let needed = total - have + 200_000_000
        guard free >= needed else {
            throw CocoaError(.fileWriteOutOfSpace, userInfo: [
                NSLocalizedDescriptionKey: "Not enough space: the model needs \(needed / 1_000_000) MB free.",
            ])
        }
        progress(have, total)
        try fm.createDirectory(at: staging, withIntermediateDirectories: true)
        AppPaths.excludeFromBackup(staging)

        for file in todo {
            try Task.checkCancellation()
            let final = staging.appending(path: file.path)
            let part = final.appendingPathExtension("download")
            try fm.createDirectory(at: final.deletingLastPathComponent(), withIntermediateDirectories: true)
            let base = have
            try await fetch(source(file.path), to: part) { progress(base + $0, total) }
            guard try sha256(of: part) == file.sha256 else {
                try? fm.removeItem(at: part)
                throw CocoaError(.fileReadCorruptFile, userInfo: [
                    NSLocalizedDescriptionKey: "\(file.path) did not match its checksum. Try again.",
                ])
            }
            try? fm.removeItem(at: final)
            try fm.moveItem(at: part, to: final)
            have += file.size
            progress(have, total)
        }

        try? fm.removeItem(at: bundle)
        try fm.moveItem(at: staging, to: bundle)
        AppPaths.excludeFromBackup(bundle)
    }

    /// One file, with byte progress. Cancelling the calling task cancels the transfer.
    private static func fetch(_ url: URL, to dest: URL, bytes: @escaping @Sendable (Int64) -> Void) async throws {
        final class Box: @unchecked Sendable { var task: URLSessionDownloadTask?; var observation: NSKeyValueObservation? }
        let box = Box()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (done: CheckedContinuation<Void, Error>) in
                let task = URLSession.shared.downloadTask(with: url) { tmp, response, error in
                    box.observation = nil
                    do {
                        if let error { throw error }
                        try check(response)
                        guard let tmp else { throw URLError(.cannotOpenFile) }
                        try? FileManager.default.removeItem(at: dest)
                        try FileManager.default.moveItem(at: tmp, to: dest)
                        done.resume()
                    } catch {
                        done.resume(throwing: error)
                    }
                }
                box.observation = task.progress.observe(\.completedUnitCount) { p, _ in bytes(p.completedUnitCount) }
                box.task = task
                task.resume()
            }
        } onCancel: {
            box.task?.cancel()
        }
    }

    private static func check(_ response: URLResponse?) throws {
        if let response, !(response is HTTPURLResponse) { return }  // file://: no status to check
        guard let http = response as? HTTPURLResponse, (200 ..< 300).contains(http.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw URLError(.badServerResponse, userInfo: [NSLocalizedDescriptionKey: "The model server answered HTTP \(code)."])
        }
    }

    private static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 8 << 20), !chunk.isEmpty { hasher.update(data: chunk) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
