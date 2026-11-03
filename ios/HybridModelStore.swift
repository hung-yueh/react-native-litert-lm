//
//  HybridModelStore.swift
//  react-native-litert-lm
//
//  Created by Antigravity on 2026-06-04.
//  Copyright © 2026 Margelo. All rights reserved.
//

import CryptoKit
import Foundation
import NitroModules

public class HybridModelStore: HybridModelStoreSpec_base, HybridModelStoreSpec_protocol {

    /// One queue for every store in the process (HybridLiteRTLM's downloadModel
    /// has a store of its own); `running` is read and written only on it.
    private static let queue = DispatchQueue(label: "dev.litert.modelstore", qos: .utility)
    /// The file names with a download running, each with the calls waiting to
    /// start after it: one download of a file at a time, so no two write
    /// `<file>.tmp` at once.
    private static var running: [String: [() -> Void]] = [:]
    /// A partial download nothing has written to for this long goes when
    /// another download starts, or when its own continuation fails.
    static let partialMaxAge: TimeInterval = 7 * 24 * 60 * 60
    /// Next to a model: the sha256 a download checked it against, with its
    /// size and modification time.
    static let verifiedSuffix = ".sha256"
    /// The cache keeps partial downloads and verified hashes next to the
    /// models under these suffixes.
    private static let reservedSuffixes = [".tmp", ResumableDownload.recordSuffix, verifiedSuffix]

    private var modelsDirectory: String {
        let cachesDir = NSSearchPathForDirectoriesInDomains(.cachesDirectory, .userDomainMask, true).first ?? NSTemporaryDirectory()
        return (cachesDir as NSString).appendingPathComponent("litert_models")
    }

    private func sanitizeFileName(_ fileName: String) throws {
        if fileName.contains("..") || fileName.contains("/") || fileName.contains("\\") {
            throw NSError(
                domain: "LiteRTLM.ModelStore",
                code: 400,
                userInfo: [NSLocalizedDescriptionKey: "Invalid filename: path traversal or directory separators are not allowed."]
            )
        }
        if fileName.isEmpty || fileName == "." || Self.reservedSuffixes.contains(where: { fileName.hasSuffix($0) }) {
            throw NSError(
                domain: "LiteRTLM.ModelStore",
                code: 400,
                userInfo: [NSLocalizedDescriptionKey:
                    "Invalid filename \"\(fileName)\": the cache reserves names ending in \(Self.reservedSuffixes.joined(separator: ", "))."]
            )
        }
    }

    public func isCached(fileName: String) throws -> Bool {
        try sanitizeFileName(fileName)
        let filePath = (modelsDirectory as NSString).appendingPathComponent(fileName)
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: filePath) {
            let attrs = try? fileManager.attributesOfItem(atPath: filePath)
            if let fileSize = attrs?[.size] as? UInt64, fileSize > 0 {
                return true
            }
        }
        return false
    }

    public func getFilePath(fileName: String) throws -> String {
        try sanitizeFileName(fileName)
        return (modelsDirectory as NSString).appendingPathComponent(fileName)
    }

    public func getFileSizeBytes(absolutePath: String) throws -> Double {
        let attrs = try? FileManager.default.attributesOfItem(atPath: absolutePath)
        if let size = attrs?[.size] as? UInt64 {
            return Double(size)
        }
        return -1.0
    }

    public func listCachedFiles() throws -> [ModelFile] {
        let fileManager = FileManager.default
        let dir = modelsDirectory

        guard fileManager.fileExists(atPath: dir) else {
            return []
        }

        let contents = try fileManager.contentsOfDirectory(atPath: dir)
        var files: [ModelFile] = []

        for name in contents {
            if Self.reservedSuffixes.contains(where: { name.hasSuffix($0) }) { continue }
            let path = (dir as NSString).appendingPathComponent(name)
            var isDir: ObjCBool = false
            if fileManager.fileExists(atPath: path, isDirectory: &isDir), !isDir.boolValue {
                let attrs = try? fileManager.attributesOfItem(atPath: path)
                let size = (attrs?[.size] as? UInt64) ?? 0
                let modDate = (attrs?[.modificationDate] as? Date) ?? Date()

                files.append(ModelFile(
                    fileName: name,
                    absolutePath: path,
                    sizeBytes: Double(size),
                    lastModifiedMs: modDate.timeIntervalSince1970 * 1000.0
                ))
            }
        }
        return files
    }

    public func deleteFile(fileName: String) throws {
        try sanitizeFileName(fileName)
        let filePath = (modelsDirectory as NSString).appendingPathComponent(fileName)
        let fileManager = FileManager.default
        // A partial download of the file goes too, unless a download is
        // writing it right now. On the store queue, without waiting for it:
        // it runs before any download of the file that starts later.
        Self.queue.async {
            if Self.running[fileName] == nil {
                try? fileManager.removeItem(atPath: filePath + ".tmp")
                try? fileManager.removeItem(atPath: filePath + ResumableDownload.recordSuffix)
            }
        }
        try? fileManager.removeItem(atPath: filePath + Self.verifiedSuffix)
        if fileManager.fileExists(atPath: filePath) {
            try fileManager.removeItem(atPath: filePath)
        }
    }

    public func downloadFile(
        url: String,
        fileName: String,
        headersJson: String,
        sha256: String,
        onProgress: @escaping (_ progress: Double) -> Void
    ) throws -> Promise<String> {
        let promise = Promise<String>()

        Self.queue.async {
            // A partial download that is never retried would stay on disk for
            // good: the ones nobody has written to for a week go now.
            self.expireStalePartials(except: fileName)
            // One download of a file at a time in this process: a call that
            // finds one running starts when it ends, then finds the file in the
            // cache or continues the partial file it left.
            Self.whenIdle(fileName) { done in
                self.download(url: url, fileName: fileName, headersJson: headersJson,
                              sha256: sha256, onProgress: onProgress) { result in
                    done()
                    switch result {
                    case .success(let path): promise.resolve(withResult: path)
                    case .failure(let error): promise.reject(withError: error)
                    }
                }
            }
        }

        return promise
    }

    /// On `queue`: runs `start` now when no download of `fileName` is running,
    /// else after the ones already waiting for it; `start` calls `done` when
    /// its download ends.
    private static func whenIdle(_ fileName: String, start: @escaping (_ done: @escaping () -> Void) -> Void) {
        let done = {
            Self.queue.async {
                if let next = Self.running[fileName]?.first {
                    Self.running[fileName]?.removeFirst()
                    next()
                } else {
                    Self.running[fileName] = nil
                }
            }
        }
        if Self.running[fileName] == nil {
            Self.running[fileName] = []
            start(done)
        } else {
            NSLog("[LiteRTLM] Waiting for the download of %@ already running", fileName)
            Self.running[fileName]?.append { start(done) }
        }
    }

    /// On `queue`: deletes the partial downloads (`<file>.tmp` and
    /// `<file>.tmp.json`) that nothing has written to for `partialMaxAge`,
    /// except `fileName`'s own (its download continues it, or starts over when
    /// the server's version changed) and those of downloads running or waiting
    /// (`running` has them). A pair is as old as the newer of its files: `.tmp`
    /// changes with every write, `.tmp.json` with every sync. A failure here
    /// is logged and does not stop the download.
    private func expireStalePartials(except fileName: String) {
        let fileManager = FileManager.default
        let dir = modelsDirectory
        guard let names = try? fileManager.contentsOfDirectory(atPath: dir) else { return }
        var lastWritten: [String: Date] = [:]
        var paths: [String: [String]] = [:]
        for name in names {
            let base: String
            if name.hasSuffix(ResumableDownload.recordSuffix) {
                base = String(name.dropLast(ResumableDownload.recordSuffix.count))
            } else if name.hasSuffix(".tmp") {
                base = String(name.dropLast(".tmp".count))
            } else {
                continue
            }
            let path = (dir as NSString).appendingPathComponent(name)
            guard base != fileName, Self.running[base] == nil,
                  let attrs = try? fileManager.attributesOfItem(atPath: path),
                  attrs[.type] as? FileAttributeType == .typeRegular,
                  let modified = attrs[.modificationDate] as? Date else { continue }
            paths[base, default: []].append(path)
            lastWritten[base] = max(lastWritten[base] ?? .distantPast, modified)
        }
        for (base, written) in lastWritten {
            let age = Date().timeIntervalSince(written)
            guard age >= Self.partialMaxAge else { continue }
            for path in paths[base] ?? [] {
                do {
                    try fileManager.removeItem(atPath: path)
                } catch {
                    NSLog("[LiteRTLM] Could not delete %@: %@", path, error.localizedDescription)
                }
            }
            NSLog("[LiteRTLM] Deleted the partial download of %@: last written %ld days ago", base, Int(age / 86_400))
        }
    }

    /// One download into the cache, or the file already there; runs on `queue`
    /// and calls `completion` once.
    private func download(
        url: String,
        fileName: String,
        headersJson: String,
        sha256: String,
        onProgress: @escaping (_ progress: Double) -> Void,
        completion: @escaping (Result<String, Error>) -> Void
    ) {
        do {
            try self.sanitizeFileName(fileName)
            let destPath = (self.modelsDirectory as NSString).appendingPathComponent(fileName)
            let fileManager = FileManager.default

            if !fileManager.fileExists(atPath: self.modelsDirectory) {
                try fileManager.createDirectory(atPath: self.modelsDirectory, withIntermediateDirectories: true, attributes: nil)
            }

            // Fast cache check. With a sha256 the cached file must be the one
            // it names: another revision under the same name is downloaded again.
            if fileManager.fileExists(atPath: destPath) {
                let attrs = try fileManager.attributesOfItem(atPath: destPath)
                if let fileSize = attrs[.size] as? UInt64, fileSize > 0 {
                    if sha256.isEmpty {
                        onProgress(1.0)
                        completion(.success(destPath))
                        return
                    }
                    // Hashing a multi-GB file must not hold the queue every store shares
                    DispatchQueue.global(qos: .utility).async {
                        if Self.matchesSha256(destPath, sha256) {
                            onProgress(1.0)
                            completion(.success(destPath))
                            return
                        }
                        NSLog("[LiteRTLM] The cached %@ is not the file sha256 %@ names: downloading it again", fileName, sha256)
                        Self.queue.async {
                            self.fetch(url: url, destPath: destPath, headersJson: headersJson, sha256: sha256,
                                       onProgress: onProgress, completion: completion)
                        }
                    }
                    return
                }
            }

            fetch(url: url, destPath: destPath, headersJson: headersJson, sha256: sha256,
                  onProgress: onProgress, completion: completion)
        } catch {
            completion(.failure(error))
        }
    }

    /// The download itself, into `<file>.tmp`, then moved over `destPath`.
    private func fetch(
        url: String,
        destPath: String,
        headersJson: String,
        sha256: String,
        onProgress: @escaping (_ progress: Double) -> Void,
        completion: @escaping (Result<String, Error>) -> Void
    ) {
        guard let downloadUrl = URL(string: url), downloadUrl.scheme?.lowercased() == "https" else {
            completion(.failure(NSError(domain: "LiteRTLM.ModelStore", code: 400, userInfo: [NSLocalizedDescriptionKey: "Invalid download URL: HTTPS is required for security."])))
            return
        }

        onProgress(0.0)

        // Parse headers JSON
        var customHeaders: [String: String] = [:]
        if !headersJson.isEmpty {
            if let data = headersJson.data(using: .utf8),
               let parsed = try? JSONSerialization.jsonObject(with: data, options: []) as? [String: String] {
                customHeaders = parsed
            }
        }

        let fileManager = FileManager.default
        let tempDestFile = destPath + ".tmp"
        let download = ResumableDownload(
            url: downloadUrl,
            headers: customHeaders,
            tempPath: tempDestFile,
            recordPath: destPath + ResumableDownload.recordSuffix,
            expectedSha256: sha256,
            onProgress: onProgress
        )
        download.start { error in
            if let error = error {
                completion(.failure(error))
                return
            }
            // An old hash must never vouch for the new file
            try? fileManager.removeItem(atPath: destPath + Self.verifiedSuffix)
            // rename(2) puts the new file in place of the old one in one step
            guard rename(tempDestFile, destPath) == 0 else {
                let code = errno
                completion(.failure(NSError(domain: NSPOSIXErrorDomain, code: Int(code), userInfo: [
                    NSLocalizedDescriptionKey: "Could not move the download into the cache: \(String(cString: strerror(code)))",
                ])))
                return
            }
            try? fileManager.removeItem(atPath: destPath + ResumableDownload.recordSuffix)
            if !sha256.isEmpty {
                Self.writeVerified(destPath, sha256: sha256)
            }

            onProgress(1.0)
            completion(.success(destPath))
        }
    }

    /// Whether the file at `path` is the one `sha256` names. The hash a
    /// download checked is kept in `<file>.sha256` with the file's size and
    /// modification time. A file without one (cached before 0.8.2, or
    /// installed without a sha256), or one changed since, is hashed once and
    /// the result kept. Off `queue`: hashing a model takes seconds.
    private static func matchesSha256(_ path: String, _ sha256: String) -> Bool {
        guard case let (size, modified)? = sizeAndModified(path) else { return false }
        if let data = FileManager.default.contents(atPath: path + verifiedSuffix),
           let kept = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           (kept["size"] as? NSNumber)?.int64Value == size,
           (kept["modified"] as? NSNumber)?.int64Value == modified {
            return kept["sha256"] as? String == sha256
        }
        let started = Date()
        var hasher = SHA256()
        do {
            try ResumableDownload.hash(path, upTo: size, into: &hasher)
        } catch {
            NSLog("[LiteRTLM] Could not hash the cached %@: %@", path, error.localizedDescription)
            return false
        }
        let actual = ResumableDownload.hex(hasher.finalize())
        NSLog("[LiteRTLM] Hashed the cached %@ in %.0f ms",
              (path as NSString).lastPathComponent, Date().timeIntervalSince(started) * 1000)
        writeVerified(path, sha256: actual)
        return actual == sha256
    }

    private static func writeVerified(_ path: String, sha256: String) {
        guard case let (size, modified)? = sizeAndModified(path),
              let data = try? JSONSerialization.data(withJSONObject: ["sha256": sha256, "size": size, "modified": modified])
        else { return }
        do {
            try data.write(to: URL(fileURLWithPath: path + verifiedSuffix))
        } catch {
            NSLog("[LiteRTLM] Could not record the sha256 of %@: %@", path, error.localizedDescription)
        }
    }

    /// The file's size, and its modification time in milliseconds.
    private static func sizeAndModified(_ path: String) -> (Int64, Int64)? {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: path),
              let size = (attrs[.size] as? NSNumber)?.int64Value,
              let modified = attrs[.modificationDate] as? Date else { return nil }
        return (size, Int64((modified.timeIntervalSince1970 * 1000).rounded()))
    }
}

/// One model download into `<file>.tmp`. When the server names the file's
/// version (a strong ETag, else Last-Modified) and size, those and the URL are
/// kept next to it in `<file>.tmp.json`, and a later download of the same URL
/// continues the partial file with a Range request instead of starting over.
/// A data task (not a download task) so the bytes land in that file as they
/// arrive and survive the app being killed. The file is synced to disk every
/// `syncBytes` and the record says how far: a power cut loses what came after,
/// never corrupts it.
final class ResumableDownload: NSObject, URLSessionDataDelegate {
    static let recordSuffix = ".tmp.json"
    /// URLProtocol classes put in front of the network (the tests' stub server).
    static var protocolClasses: [AnyClass] = []
    /// A partial download is synced to disk, and its record updated, every
    /// this many bytes. Tests shorten it.
    static var syncBytes: Int64 = 64 << 20
    /// Room a download leaves free on top of the file.
    static let freeSpaceMargin: Int64 = 64 << 20
    /// Free bytes on the volume holding a directory. Tests replace it.
    static var availableCapacity: (String) -> Int64 = { dir in
        let url = URL(fileURLWithPath: dir)
        if let capacity = (try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
            .volumeAvailableCapacityForImportantUsage {
            return capacity
        }
        return ((try? FileManager.default.attributesOfFileSystem(forPath: dir))?[.systemFreeSize] as? NSNumber)?
            .int64Value ?? .max
    }

    private let url: URL
    private let headers: [String: String]
    private let tempPath: String
    private let recordPath: String
    private let expectedSha256: String
    private let onProgress: (Double) -> Void
    private var completion: ((Error?) -> Void)?

    private var session: URLSession!
    private var record: [String: Any]?
    private var offset: Int64 = 0
    private var totalBytes: Int64 = -1
    private var expectedLength: Int64 = -1
    private var received: Int64 = 0
    private var synced: Int64 = 0
    private var identityEncoded = true
    private var keepPartial = false
    private var stale = false
    private var startOver = false
    private var failure: Error?
    private var file: FileHandle?
    private var hasher: SHA256?
    private var lastUpdate = Date.distantPast

    init(url: URL, headers: [String: String], tempPath: String, recordPath: String,
         expectedSha256: String, onProgress: @escaping (Double) -> Void) {
        self.url = url
        self.headers = headers
        self.tempPath = tempPath
        self.recordPath = recordPath
        self.expectedSha256 = expectedSha256
        self.onProgress = onProgress
    }

    /// Calls `completion` once: nil when `<file>.tmp` holds the whole file
    /// (and matches the expected sha256), else the error.
    func start(completion: @escaping (Error?) -> Void) {
        self.completion = completion
        let sessionConfig = URLSessionConfiguration.default
        sessionConfig.timeoutIntervalForRequest = 30
        // A multi-GB model on a slow connection: an hour cut those short
        sessionConfig.timeoutIntervalForResource = 24 * 60 * 60
        if !Self.protocolClasses.isEmpty {
            sessionConfig.protocolClasses = Self.protocolClasses + (sessionConfig.protocolClasses ?? [])
        }
        let delegateQueue = OperationQueue()
        delegateQueue.maxConcurrentOperationCount = 1
        session = URLSession(configuration: sessionConfig, delegate: self, delegateQueue: delegateQueue)

        // A partial file left by an earlier call is continued only when it came
        // from the same URL (its query aside: a signed URL changes with every
        // call) and that response named its version and size. Only the bytes
        // synced to disk count: after a power cut the ones after them may never
        // have been written.
        let size = Self.fileSize(tempPath)
        if let data = FileManager.default.contents(atPath: recordPath),
           let saved = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           saved["url"] as? String == Self.withoutQuery(url),
           saved["validator"] is String,
           let total = (saved["totalBytes"] as? NSNumber)?.int64Value,
           let syncedBytes = (saved["syncedBytes"] as? NSNumber)?.int64Value,
           syncedBytes > 0, syncedBytes < total, size >= syncedBytes {
            record = saved
            offset = syncedBytes
            totalBytes = total
            keepPartial = true
            // Before anything writes to the pair: a partial that nothing could
            // add to for a week goes when its continuation fails
            stale = Date().timeIntervalSince(Self.lastWritten(tempPath, recordPath)) >= HybridModelStore.partialMaxAge
        } else {
            discardPartial()
        }

        // On this download's own queue (not the one every store shares): the
        // part on disk is hashed before the request, so the response doesn't
        // wait on it
        delegateQueue.addOperation {
            if self.offset > 0 {
                self.onProgress(Double(self.offset) / Double(self.totalBytes))
                if !self.expectedSha256.isEmpty {
                    var hasher = SHA256()
                    do {
                        let started = Date()
                        try Self.hash(self.tempPath, upTo: self.offset, into: &hasher)
                        self.hasher = hasher
                        NSLog("[LiteRTLM] Hashed the %lld bytes already downloaded in %.0f ms",
                              self.offset, Date().timeIntervalSince(started) * 1000)
                    } catch {
                        NSLog("[LiteRTLM] Could not read the partial download (%@): starting over",
                              error.localizedDescription)
                        self.discardPartial()
                        self.record = nil
                        self.offset = 0
                        self.totalBytes = -1
                        self.keepPartial = false
                    }
                }
            }
            self.session.dataTask(with: self.request()).resume()
        }
    }

    private func request() -> URLRequest {
        var request = URLRequest(url: url)
        for (key, value) in headers {
            request.addValue(value, forHTTPHeaderField: key)
        }
        if offset > 0, let validator = record?["validator"] as? String {
            request.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range")
            request.setValue(validator, forHTTPHeaderField: "If-Range")
        }
        return request
    }

    // The range must reach the server the URL redirects to (Hugging Face
    // redirects to its CDN), and every hop must be HTTPS, not just the first.
    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        guard newRequest.url?.scheme?.lowercased() == "https" else {
            failure = Self.error(400, "Refusing to follow a redirect to a URL that is not HTTPS.")
            completionHandler(nil)
            return
        }
        var redirected = newRequest
        for field in ["Range", "If-Range"] {
            if let value = task.originalRequest?.value(forHTTPHeaderField: field),
               redirected.value(forHTTPHeaderField: field) == nil {
                redirected.setValue(value, forHTTPHeaderField: field)
            }
        }
        completionHandler(redirected)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        guard failure == nil else {
            completionHandler(.cancel)
            return
        }
        guard let http = response as? HTTPURLResponse else {
            failure = Self.error(500, "No HTTP response.")
            completionHandler(.cancel)
            return
        }
        let code = http.statusCode
        if offset > 0 && !continues(http) {
            // 200 = the whole file (the server ignored the range, or the file
            // changed); 416 or a 206 of another version = the partial file is
            // no use. Anything else fails and keeps it for next time, unless
            // the file is gone (404, 410) or the partial has gone a week
            // without growing: in an app with one model, no other download's
            // clean-up would ever delete it.
            guard code == 200 || code == 206 || code == 416 else {
                if code == 404 || code == 410 || stale {
                    keepPartial = false
                }
                failure = Self.error(code, "HTTP \(code)")
                completionHandler(.cancel)
                return
            }
            NSLog("[LiteRTLM] Not continuing the partial download (HTTP %ld): starting over", code)
            discardPartial()
            record = nil
            offset = 0
            totalBytes = -1
            keepPartial = false
            if code != 200 {
                startOver = true
                completionHandler(.cancel)
                return
            }
        }
        // A new download takes a 200 only: a 206 to a request without a Range
        // (or any other answer) is not the whole file.
        if offset == 0 && code != 200 {
            failure = Self.error(code, "HTTP \(code)")
            completionHandler(.cancel)
            return
        }

        do {
            // A continuation's hasher holds the part on disk already
            if offset == 0 && !expectedSha256.isEmpty {
                hasher = SHA256()
            }
            expectedLength = http.expectedContentLength
            let encoding = http.value(forHTTPHeaderField: "Content-Encoding")
            identityEncoded = encoding == nil || encoding?.lowercased() == "identity"
            // Not one byte is written without room for the rest of the file: a
            // download that fills the disk would leave it full
            let total = offset > 0 ? totalBytes : expectedLength
            if total > 0 {
                try Self.requireFreeSpace((tempPath as NSString).deletingLastPathComponent, total - offset)
            }
            if offset > 0 {
                NSLog("[LiteRTLM] Continuing the partial download at byte %lld of %lld", offset, totalBytes)
                let handle = try FileHandle(forWritingTo: URL(fileURLWithPath: tempPath))
                try handle.truncate(atOffset: UInt64(offset))
                try handle.seekToEnd()
                file = handle
            } else {
                FileManager.default.createFile(atPath: tempPath, contents: nil)
                file = try FileHandle(forWritingTo: URL(fileURLWithPath: tempPath))
                totalBytes = expectedLength
                // What the next call needs to continue this download
                if let validator = Self.validator(of: http), expectedLength > 0, identityEncoded {
                    record = ["url": Self.withoutQuery(url), "validator": validator, "totalBytes": expectedLength]
                    try writeRecord(synced: 0)
                    keepPartial = true
                }
            }
            synced = offset
            completionHandler(.allow)
        } catch {
            failure = error
            completionHandler(.cancel)
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        do {
            try file?.write(contentsOf: data)
            hasher?.update(data: data)
            received += Int64(data.count)
            if record != nil && offset + received - synced >= Self.syncBytes {
                try file?.synchronize()
                synced = offset + received
                try writeRecord(synced: synced)
            }
        } catch {
            // A disk the download filled must not stay full
            if Self.isOutOfSpace(error) {
                keepPartial = false
            }
            failure = error
            try? file?.close()
            file = nil
            dataTask.cancel()
            return
        }
        if totalBytes > 0 {
            let now = Date()
            if now.timeIntervalSince(lastUpdate) > 0.1 {
                onProgress(Double(offset + received) / Double(totalBytes))
                lastUpdate = now
            }
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        // However the transfer ends, what it wrote reaches the disk before the
        // record says so, or before the file is installed
        if let file = file {
            do {
                try file.synchronize()
                if record != nil {
                    try writeRecord(synced: offset + received)
                }
            } catch {
                NSLog("[LiteRTLM] Could not sync the download to disk: %@", error.localizedDescription)
            }
            try? file.close()
            self.file = nil
        }
        if startOver {
            startOver = false
            received = 0
            session.dataTask(with: request()).resume()
            return
        }
        if let failure = failure ?? error {
            finish(failure)
            return
        }
        // A dropped connection can end the response early without an error —
        // never install a truncated model file. A continued download must
        // reach the size its first response gave: a 206 sent chunked has no
        // Content-Length to compare with.
        if offset > 0 && offset + received != totalBytes {
            finish(Self.error(500, "Incomplete download: got \(offset + received) of \(totalBytes) bytes"))
            return
        }
        if offset == 0 && identityEncoded && expectedLength > 0 && received != expectedLength {
            finish(Self.error(500, "Incomplete download: got \(received) of \(expectedLength) bytes"))
            return
        }
        NSLog("[LiteRTLM] Received %lld bytes from byte %lld: %lld of %lld bytes",
              received, offset, offset + received, totalBytes)
        if let hasher = hasher {
            let started = Date()
            let actual = Self.hex(hasher.finalize())
            if actual != expectedSha256 {
                keepPartial = false
                finish(Self.error(500, "sha256 mismatch: expected \(expectedSha256), got \(actual)"))
                return
            }
            NSLog("[LiteRTLM] sha256 matches (%@), final digest in %.0f ms",
                  actual, Date().timeIntervalSince(started) * 1000)
        }
        finish(nil)
    }

    private func finish(_ error: Error?) {
        if error != nil {
            if keepPartial && Self.fileSize(tempPath) > 0 {
                NSLog("[LiteRTLM] Kept %lld bytes of the partial download for the next call", Self.fileSize(tempPath))
            } else {
                discardPartial()
            }
        }
        session.finishTasksAndInvalidate()
        let done = completion
        completion = nil
        done?(error)
    }

    /// A 206 with the rest of the file the partial download holds. The
    /// validator is compared here too: some servers answer a Range request
    /// without evaluating If-Range (the Hugging Face CDN does).
    private func continues(_ response: HTTPURLResponse) -> Bool {
        guard response.statusCode == 206,
              let range = response.value(forHTTPHeaderField: "Content-Range"),
              range.hasPrefix("bytes ") else { return false }
        let parts = range.dropFirst(6).split(whereSeparator: { $0 == "-" || $0 == "/" })
        guard parts.count == 3,
              let first = Int64(parts[0]), let last = Int64(parts[1]), let size = Int64(parts[2]) else { return false }
        return first == offset && last == totalBytes - 1 && size == totalBytes
            && Self.validator(of: response) == record?["validator"] as? String
    }

    private func writeRecord(synced: Int64) throws {
        guard var saved = record else { return }
        saved["syncedBytes"] = synced
        try JSONSerialization.data(withJSONObject: saved).write(to: URL(fileURLWithPath: recordPath))
    }

    private func discardPartial() {
        try? FileManager.default.removeItem(atPath: tempPath)
        try? FileManager.default.removeItem(atPath: recordPath)
    }

    /// Feeds the first `length` bytes of the file at `path` to `hasher`, 1 MiB at a time.
    static func hash(_ path: String, upTo length: Int64, into hasher: inout SHA256) throws {
        guard let input = FileHandle(forReadingAtPath: path) else {
            throw Self.error(500, "Cannot read \((path as NSString).lastPathComponent).")
        }
        defer { try? input.close() }
        var left = length
        while left > 0 {
            // One pool per chunk: a multi-GB file is read in one go
            let count = try autoreleasepool { () -> Int in
                guard let chunk = try input.read(upToCount: Int(min(left, 1 << 20))), !chunk.isEmpty else {
                    throw Self.error(500, "\((path as NSString).lastPathComponent) is shorter than \(length) bytes.")
                }
                hasher.update(data: chunk)
                return chunk.count
            }
            left -= Int64(count)
        }
    }

    static func hex(_ digest: SHA256.Digest) -> String {
        digest.map { String(format: "%02x", $0) }.joined()
    }

    private static func requireFreeSpace(_ dir: String, _ needed: Int64) throws {
        let free = availableCapacity(dir)
        if free < needed + freeSpaceMargin {
            throw Self.error(507, "Not enough free storage: \(needed >> 20) MB left to download, \(free >> 20) MB free")
        }
    }

    private static func isOutOfSpace(_ error: Error) -> Bool {
        var current: NSError? = error as NSError
        while let nsError = current {
            if (nsError.domain == NSCocoaErrorDomain && nsError.code == NSFileWriteOutOfSpaceError)
                || (nsError.domain == NSPOSIXErrorDomain && nsError.code == Int(ENOSPC)) {
                return true
            }
            current = nsError.userInfo[NSUnderlyingErrorKey] as? NSError
        }
        return false
    }

    /// The URL a partial download is recorded under: a signed URL's query changes with every call.
    private static func withoutQuery(_ url: URL) -> String {
        let string = url.absoluteString
        return String(string[..<(string.firstIndex(where: { $0 == "?" || $0 == "#" }) ?? string.endIndex)])
    }

    /// The version a response is of: a strong ETag, else Last-Modified (what If-Range accepts).
    private static func validator(of response: HTTPURLResponse) -> String? {
        if let etag = response.value(forHTTPHeaderField: "ETag"), !etag.hasPrefix("W/") {
            return etag
        }
        return response.value(forHTTPHeaderField: "Last-Modified")
    }

    private static func fileSize(_ path: String) -> Int64 {
        ((try? FileManager.default.attributesOfItem(atPath: path))?[.size] as? NSNumber)?.int64Value ?? 0
    }

    /// When the newer of two files was last written.
    private static func lastWritten(_ paths: String...) -> Date {
        paths.compactMap { (try? FileManager.default.attributesOfItem(atPath: $0))?[.modificationDate] as? Date }
            .max() ?? .distantPast
    }

    private static func error(_ code: Int, _ message: String) -> NSError {
        NSError(domain: "LiteRTLM.ModelStore", code: code, userInfo: [NSLocalizedDescriptionKey: message])
    }
}
