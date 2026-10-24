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
            if name.hasSuffix(".tmp") || name.hasSuffix(ResumableDownload.recordSuffix) { continue }
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
        // writing it right now
        Self.queue.sync {
            if Self.running[fileName] == nil {
                try? fileManager.removeItem(atPath: filePath + ".tmp")
                try? fileManager.removeItem(atPath: filePath + ResumableDownload.recordSuffix)
            }
        }
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
            
            // Fast cache check
            if fileManager.fileExists(atPath: destPath) {
                let attrs = try fileManager.attributesOfItem(atPath: destPath)
                if let fileSize = attrs[.size] as? UInt64, fileSize > 0 {
                    onProgress(1.0)
                    completion(.success(destPath))
                    return
                }
            }
            
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
            
            let tempDestFile = (self.modelsDirectory as NSString).appendingPathComponent("\(fileName).tmp")
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
                do {
                    if fileManager.fileExists(atPath: destPath) {
                        try fileManager.removeItem(atPath: destPath)
                    }
                    try fileManager.moveItem(at: URL(fileURLWithPath: tempDestFile), to: URL(fileURLWithPath: destPath))
                    try? fileManager.removeItem(atPath: destPath + ResumableDownload.recordSuffix)

                    onProgress(1.0)
                    completion(.success(destPath))
                } catch {
                    completion(.failure(error))
                }
            }
        } catch {
            completion(.failure(error))
        }
    }
}

/// One model download into `<file>.tmp`. When the server names the file's
/// version (a strong ETag, else Last-Modified) and size, those and the URL are
/// kept next to it in `<file>.tmp.json`, and a later download of the same URL
/// continues the partial file with a Range request instead of starting over.
/// A data task (not a download task) so the bytes land in that file as they
/// arrive and survive the app being killed.
final class ResumableDownload: NSObject, URLSessionDataDelegate {
    static let recordSuffix = ".tmp.json"
    /// URLProtocol classes put in front of the network (the tests' stub server).
    static var protocolClasses: [AnyClass] = []

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
    private var identityEncoded = true
    private var keepPartial = false
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
        sessionConfig.timeoutIntervalForResource = 3600
        if !Self.protocolClasses.isEmpty {
            sessionConfig.protocolClasses = Self.protocolClasses + (sessionConfig.protocolClasses ?? [])
        }
        let delegateQueue = OperationQueue()
        delegateQueue.maxConcurrentOperationCount = 1
        session = URLSession(configuration: sessionConfig, delegate: self, delegateQueue: delegateQueue)

        // A partial file left by an earlier call is continued only when it came
        // from the same URL and that response named its version and size.
        let size = Self.fileSize(tempPath)
        if let data = FileManager.default.contents(atPath: recordPath),
           let saved = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           saved["url"] as? String == url.absoluteString,
           saved["validator"] is String,
           let total = (saved["totalBytes"] as? NSNumber)?.int64Value,
           size > 0, size < total {
            record = saved
            offset = size
            totalBytes = total
            keepPartial = true
        } else {
            discardPartial()
        }
        session.dataTask(with: request()).resume()
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
    // redirects to its CDN).
    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
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
        guard let http = response as? HTTPURLResponse else {
            failure = Self.error(500, "No HTTP response.")
            completionHandler(.cancel)
            return
        }
        let code = http.statusCode
        if offset > 0 && !continues(http) {
            // 200 = the whole file (the server ignored the range, or the file
            // changed); 416 or a 206 of another version = the partial file is
            // no use. Anything else fails and keeps it for next time.
            guard code == 200 || code == 206 || code == 416 else {
                failure = Self.error(code, "HTTP \(code)")
                completionHandler(.cancel)
                return
            }
            NSLog("[LiteRTLM] Not continuing the partial download (HTTP %ld): starting over", code)
            discardPartial()
            offset = 0
            totalBytes = -1
            keepPartial = false
            if code != 200 {
                startOver = true
                completionHandler(.cancel)
                return
            }
        }
        if offset == 0 && code >= 400 {
            failure = Self.error(code, "HTTP \(code)")
            completionHandler(.cancel)
            return
        }

        do {
            if !expectedSha256.isEmpty {
                hasher = SHA256()
            }
            expectedLength = http.expectedContentLength
            let encoding = http.value(forHTTPHeaderField: "Content-Encoding")
            identityEncoded = encoding == nil || encoding?.lowercased() == "identity"
            if offset > 0 {
                NSLog("[LiteRTLM] Continuing the partial download at byte %lld of %lld", offset, totalBytes)
                if hasher != nil {
                    let started = Date()
                    try hashPrefix()
                    NSLog("[LiteRTLM] Hashed the %lld bytes already downloaded in %.0f ms",
                          offset, Date().timeIntervalSince(started) * 1000)
                }
                let handle = try FileHandle(forWritingTo: URL(fileURLWithPath: tempPath))
                try handle.truncate(atOffset: UInt64(offset))
                try handle.seekToEnd()
                file = handle
                onProgress(Double(offset) / Double(totalBytes))
            } else {
                FileManager.default.createFile(atPath: tempPath, contents: nil)
                file = try FileHandle(forWritingTo: URL(fileURLWithPath: tempPath))
                totalBytes = expectedLength
                // What the next call needs to continue this download
                if let validator = Self.validator(of: http), expectedLength > 0, identityEncoded {
                    let saved: [String: Any] = [
                        "url": url.absoluteString, "validator": validator, "totalBytes": expectedLength,
                    ]
                    try JSONSerialization.data(withJSONObject: saved).write(to: URL(fileURLWithPath: recordPath))
                    keepPartial = true
                }
            }
            completionHandler(.allow)
        } catch {
            failure = error
            completionHandler(.cancel)
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        do {
            try file?.write(contentsOf: data)
        } catch {
            failure = error
            try? file?.close()
            file = nil
            dataTask.cancel()
            return
        }
        hasher?.update(data: data)
        received += Int64(data.count)
        if totalBytes > 0 {
            let now = Date()
            if now.timeIntervalSince(lastUpdate) > 0.1 {
                onProgress(Double(offset + received) / Double(totalBytes))
                lastUpdate = now
            }
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        try? file?.close()
        file = nil
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
            let actual = hasher.finalize().map { String(format: "%02x", $0) }.joined()
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

    private func hashPrefix() throws {
        guard let input = FileHandle(forReadingAtPath: tempPath) else {
            throw Self.error(500, "Cannot read the partial download.")
        }
        defer { try? input.close() }
        var left = offset
        while left > 0 {
            // One pool per chunk: a multi-GB partial file is read on one delegate callback
            let count = try autoreleasepool { () -> Int in
                guard let chunk = try input.read(upToCount: Int(min(left, 1 << 20))), !chunk.isEmpty else {
                    throw Self.error(500, "The partial download is shorter than \(offset) bytes.")
                }
                hasher?.update(data: chunk)
                return chunk.count
            }
            left -= Int64(count)
        }
    }

    private func discardPartial() {
        try? FileManager.default.removeItem(atPath: tempPath)
        try? FileManager.default.removeItem(atPath: recordPath)
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

    private static func error(_ code: Int, _ message: String) -> NSError {
        NSError(domain: "LiteRTLM.ModelStore", code: code, userInfo: [NSLocalizedDescriptionKey: message])
    }
}
