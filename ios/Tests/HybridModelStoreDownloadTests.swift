import CryptoKit
import XCTest
import NitroModules
@testable import LiteRTLM

/// What HybridModelStore's download does when the network answers, against a
/// stub server (a URLProtocol in front of the session) — the cases
/// HybridModelStoreDownloadTest covers on Android: a partial download kept and
/// continued with a Range request; the answers that start it over or keep it
/// (a server that ignores the range, a 416, a 206 of another version, a 503);
/// a continuation short of the whole file; two downloads of one file at once;
/// and the sha256 check.
final class HybridModelStoreDownloadTests: XCTestCase {
    private var store: HybridModelStore!
    private var names: [String] = []
    private let payload = Data((0..<(300 * 1024)).map { UInt8(truncatingIfNeeded: $0 &* 7 &+ 3) })

    override func setUp() {
        super.setUp()
        StubServer.reset()
        ResumableDownload.protocolClasses = [StubServer.self]
        store = HybridModelStore()
    }

    override func tearDown() {
        for name in names {
            for suffix in ["", ".tmp", ".tmp.json"] {
                try? FileManager.default.removeItem(atPath: path(name, suffix))
            }
        }
        ResumableDownload.protocolClasses = []
        StubServer.reset()
        store = nil
        super.tearDown()
    }

    // MARK: - Helpers

    /// A file name of this test; its files go in tearDown.
    private func fileName(_ base: String) -> String {
        let name = "\(base)-\(UUID().uuidString.prefix(8)).litertlm"
        names.append(name)
        return name
    }

    private func path(_ name: String, _ suffix: String = "") -> String {
        try! store.getFilePath(fileName: name) + suffix
    }

    /// Bytes of the file, nil when it is not there.
    private func size(_ name: String, _ suffix: String = "") -> Int? {
        (try? FileManager.default.attributesOfItem(atPath: path(name, suffix)))?[.size] as? Int
    }

    private func download(_ name: String, sha256: String = "", progress: ProgressLog? = nil) throws -> Promise<String> {
        try store.downloadFile(url: "https://stub.test/models/\(name)", fileName: name, headersJson: "{}",
                               sha256: sha256) { value in progress?.add(value) }
    }

    /// The promise's result, or a failure when it has not settled within `seconds`.
    private func settle(_ promise: Promise<String>, within seconds: TimeInterval = 10) -> Result<String, Error> {
        let settled = DispatchSemaphore(value: 0)
        let box = ResultBox()
        promise.then { path in
            box.result = .success(path)
            settled.signal()
        }.catch { error in
            box.result = .failure(error)
            settled.signal()
        }
        guard settled.wait(timeout: .now() + seconds) == .success, let result = box.result else {
            return .failure(NSError(domain: "test", code: 0,
                                    userInfo: [NSLocalizedDescriptionKey: "Not settled within \(seconds) s"]))
        }
        return result
    }

    /// The path the download resolved with; a rejection fails the test with the whole error.
    private func resolved(_ promise: Promise<String>, within seconds: TimeInterval = 10,
                          file: StaticString = #filePath, line: UInt = #line) throws -> String {
        switch settle(promise, within: seconds) {
        case .success(let path):
            return path
        case .failure(let error):
            XCTFail("The download rejected: \(error as NSError)", file: file, line: line)
            throw error
        }
    }

    private func waitUntil(_ what: String, _ condition: () -> Bool) {
        let deadline = Date().addingTimeInterval(5)
        while !condition() && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.01)
        }
        XCTAssertTrue(condition(), what)
    }

    private func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// The whole payload with its version and size; `pauseAfter` bytes, then it waits for `resume`.
    private func whole(etag: String = "\"v1\"", body: Data? = nil, pauseAfter: Int? = nil,
                       resume: DispatchSemaphore? = nil, dropAfter: Int? = nil) -> StubAnswer {
        let body = body ?? payload
        return StubAnswer(status: 200, headers: ["Content-Length": "\(body.count)", "ETag": etag], body: body,
                          pauseAfter: pauseAfter, resume: resume, dropAfter: dropAfter)
    }

    /// The rest of the payload from byte `from`.
    private func rest(from: Int, etag: String = "\"v1\"") -> StubAnswer {
        StubAnswer(status: 206, headers: [
            "Content-Range": "bytes \(from)-\(payload.count - 1)/\(payload.count)",
            "Content-Length": "\(payload.count - from)",
            "ETag": etag,
        ], body: payload.subdata(in: from..<payload.count))
    }

    /// A first call that the connection drops after `cut` bytes, from a server
    /// that named the file's version and size: the partial file stays. The
    /// connection drops once those bytes are on disk (a drop that comes before
    /// the session hands them over keeps nothing: testADropBeforeAnyByteIsWrittenStartsTheNextCallOver).
    private func leavePartial(_ name: String, cut: Int, sha256: String = "") throws {
        let resume = DispatchSemaphore(value: 0)
        StubServer.enqueue(whole(pauseAfter: cut, resume: resume, dropAfter: cut))
        let promise = try download(name, sha256: sha256)
        waitUntil("the download writes \(cut) bytes") { size(name, ".tmp") == cut }
        resume.signal()
        if case .success = settle(promise) {
            XCTFail("A dropped download must reject")
        }
        XCTAssertEqual(size(name, ".tmp"), cut)
        XCTAssertNotNil(size(name, ".tmp.json"))
        XCTAssertNil(size(name))
        XCTAssertEqual(StubServer.requests.count, 1)
        StubServer.forgetRequests()
    }

    // MARK: - A download and its partial file

    func testADownloadIsMovedIntoPlace() throws {
        let name = fileName("whole")
        StubServer.enqueue(whole())
        let progress = ProgressLog()

        let path = try resolved(try download(name, progress: progress))

        XCTAssertEqual(FileManager.default.contents(atPath: path), payload)
        XCTAssertNil(size(name, ".tmp"))
        XCTAssertNil(size(name, ".tmp.json"))
        XCTAssertEqual(progress.values.first, 0)
        XCTAssertEqual(progress.values.last, 1)
        XCTAssertEqual(progress.values, progress.values.sorted())
    }

    func testADroppedDownloadKeepsItsPartialFile() throws {
        let name = fileName("kept")
        try leavePartial(name, cut: 100 * 1024)

        XCTAssertFalse(try store.isCached(fileName: name))
        XCTAssertFalse(try store.listCachedFiles().contains { $0.fileName.hasPrefix(name) })
    }

    func testADropBeforeAnyByteIsWrittenStartsTheNextCallOver() throws {
        // The connection drops right after the headers: the session can end the
        // task before it hands the delegate a byte, so the partial file may be
        // empty and go. Either way, a wrong sha256 on the next call rejects the
        // whole file and leaves nothing; the call after that gets the file.
        let name = fileName("early")
        StubServer.enqueue(whole(dropAfter: 100 * 1024))
        if case .success = settle(try download(name)) {
            XCTFail("A dropped download must reject")
        }
        StubServer.enqueue(whole())
        let wrong = String(repeating: "0", count: 64)

        guard case .failure(let error) = settle(try download(name, sha256: wrong)) else {
            return XCTFail("A sha256 mismatch must reject")
        }

        XCTAssertTrue(error.localizedDescription.contains("sha256 mismatch"), "\(error as NSError)")
        XCTAssertNil(size(name))
        XCTAssertNil(size(name, ".tmp"))
        XCTAssertNil(size(name, ".tmp.json"))
        StubServer.enqueue(whole())
        let path = try resolved(try download(name, sha256: sha256(payload)))
        XCTAssertEqual(FileManager.default.contents(atPath: path), payload)
    }

    func testTheNextCallContinuesWithARangeRequest() throws {
        let name = fileName("resume")
        try leavePartial(name, cut: 100 * 1024)
        StubServer.enqueue(rest(from: 100 * 1024))
        let progress = ProgressLog()

        let path = try resolved(try download(name, progress: progress))

        XCTAssertEqual(FileManager.default.contents(atPath: path), payload)
        let request = try XCTUnwrap(StubServer.requests.first)
        XCTAssertEqual(request.value(forHTTPHeaderField: "Range"), "bytes=\(100 * 1024)-")
        XCTAssertEqual(request.value(forHTTPHeaderField: "If-Range"), "\"v1\"")
        XCTAssertNil(size(name, ".tmp"))
        XCTAssertNil(size(name, ".tmp.json"))
        // Progress counts the bytes already there, ends at 1, never decreases
        let values = progress.values
        XCTAssertGreaterThanOrEqual(values.count, 2)
        XCTAssertEqual(values.dropFirst().first ?? -1, Double(100 * 1024) / Double(payload.count), accuracy: 1e-9)
        XCTAssertEqual(values.last, 1)
        XCTAssertEqual(values, values.sorted())
    }

    func testAServerThatIgnoresTheRangeStartsTheFileOver() throws {
        let name = fileName("norange")
        try leavePartial(name, cut: 100 * 1024)
        StubServer.enqueue(whole())

        let path = try resolved(try download(name))

        XCTAssertEqual(FileManager.default.contents(atPath: path), payload)
        XCTAssertEqual(StubServer.requests.count, 1)
    }

    func testAnUnsatisfiableRangeStartsOverWithoutIt() throws {
        let name = fileName("r416")
        try leavePartial(name, cut: 100 * 1024)
        StubServer.enqueue(StubAnswer(status: 416, headers: ["Content-Range": "bytes */\(payload.count)"]))
        StubServer.enqueue(whole())

        let path = try resolved(try download(name))

        XCTAssertEqual(FileManager.default.contents(atPath: path), payload)
        let requests = StubServer.requests
        XCTAssertEqual(requests.count, 2)
        XCTAssertNotNil(requests.first?.value(forHTTPHeaderField: "Range"))
        XCTAssertNil(requests.last?.value(forHTTPHeaderField: "Range"), "The second request asks for the whole file")
    }

    func testARangeOfAnotherVersionIsNotAppended() throws {
        // The Hugging Face CDN answers 206 whatever If-Range says: the validator
        // of the 206 itself has to match.
        let name = fileName("changed")
        try leavePartial(name, cut: 100 * 1024)
        let v2 = Data(payload.reversed())
        StubServer.enqueue(rest(from: 100 * 1024, etag: "\"v2\""))
        StubServer.enqueue(whole(etag: "\"v2\"", body: v2))

        let path = try resolved(try download(name))

        XCTAssertEqual(FileManager.default.contents(atPath: path), v2)
        XCTAssertEqual(StubServer.requests.count, 2)
        XCTAssertNil(StubServer.requests.last?.value(forHTTPHeaderField: "Range"))
    }

    func testAServerErrorOnTheContinuationKeepsThePartialFile() throws {
        let name = fileName("busy")
        try leavePartial(name, cut: 100 * 1024)
        StubServer.enqueue(StubAnswer(status: 503))

        guard case .failure(let error) = settle(try download(name)) else {
            return XCTFail("A 503 must reject")
        }

        XCTAssertTrue(error.localizedDescription.contains("503"), error.localizedDescription)
        XCTAssertEqual(size(name, ".tmp"), 100 * 1024)
        XCTAssertNotNil(size(name, ".tmp.json"))
    }

    func testAContinuationShortOfTheWholeFileIsNotInstalled() throws {
        let name = fileName("short")
        try leavePartial(name, cut: 100 * 1024)
        // A 206 sent chunked (no Content-Length to check) that ends 50 KiB in
        StubServer.enqueue(StubAnswer(status: 206, headers: [
            "Content-Range": "bytes \(100 * 1024)-\(payload.count - 1)/\(payload.count)",
            "ETag": "\"v1\"",
        ], body: payload.subdata(in: (100 * 1024)..<(150 * 1024))))

        guard case .failure(let error) = settle(try download(name)) else {
            return XCTFail("A continuation short of the whole file must reject")
        }

        XCTAssertTrue(error.localizedDescription.contains("Incomplete download: got \(150 * 1024) of \(payload.count) bytes"),
                      error.localizedDescription)
        XCTAssertNil(size(name))
        // What it got stays for the next call, which continues from there
        XCTAssertEqual(size(name, ".tmp"), 150 * 1024)
        XCTAssertNotNil(size(name, ".tmp.json"))
        StubServer.enqueue(rest(from: 150 * 1024))

        let path = try resolved(try download(name))

        XCTAssertEqual(FileManager.default.contents(atPath: path), payload)
        XCTAssertEqual(StubServer.requests.last?.value(forHTTPHeaderField: "Range"), "bytes=\(150 * 1024)-")
    }

    // MARK: - Two downloads of one file

    func testTwoDownloadsOfTheSameFileRunOneAfterTheOther() throws {
        let name = fileName("twice")
        let resume = DispatchSemaphore(value: 0)
        StubServer.enqueue(whole(pauseAfter: 100 * 1024, resume: resume))

        let first = try download(name)
        waitUntil("the first download writes its partial file") { (size(name, ".tmp") ?? 0) >= 100 * 1024 }
        let second = try download(name)
        Thread.sleep(forTimeInterval: 0.5)
        XCTAssertEqual(StubServer.requests.count, 1, "The second call waits for the first instead of downloading")
        resume.signal()

        let path1 = try resolved(first)
        let path2 = try resolved(second)
        XCTAssertEqual(path1, path2)
        XCTAssertEqual(FileManager.default.contents(atPath: path1), payload)
        // The second found the file in the cache
        XCTAssertEqual(StubServer.requests.count, 1)
    }

    func testACallThatWaitedContinuesWhatAFailedDownloadLeft() throws {
        let name = fileName("again")
        let resume = DispatchSemaphore(value: 0)
        StubServer.enqueue(whole(pauseAfter: 100 * 1024, resume: resume, dropAfter: 100 * 1024))
        StubServer.enqueue(rest(from: 100 * 1024))

        let first = try download(name)
        waitUntil("the first download writes its partial file") { (size(name, ".tmp") ?? 0) >= 100 * 1024 }
        let second = try download(name)
        Thread.sleep(forTimeInterval: 0.5)
        XCTAssertEqual(StubServer.requests.count, 1)
        resume.signal()

        if case .success = settle(first) {
            XCTFail("The dropped download must reject")
        }
        let path = try resolved(second)
        XCTAssertEqual(FileManager.default.contents(atPath: path), payload)
        XCTAssertEqual(StubServer.requests.last?.value(forHTTPHeaderField: "Range"), "bytes=\(100 * 1024)-")
    }

    func testDeleteFileLeavesThePartialFileOfARunningDownload() throws {
        let name = fileName("writing")
        let resume = DispatchSemaphore(value: 0)
        StubServer.enqueue(whole(pauseAfter: 100 * 1024, resume: resume))

        let running = try download(name)
        waitUntil("the download writes its partial file") { (size(name, ".tmp") ?? 0) >= 100 * 1024 }
        try store.deleteFile(fileName: name)

        XCTAssertEqual(size(name, ".tmp"), 100 * 1024)
        XCTAssertNotNil(size(name, ".tmp.json"))
        resume.signal()
        let path = try resolved(running)
        XCTAssertEqual(FileManager.default.contents(atPath: path), payload)
    }

    func testADownloadOfAnotherFileDoesNotWait() throws {
        let slow = fileName("slow")
        let other = fileName("other")
        let resume = DispatchSemaphore(value: 0)
        StubServer.enqueue(whole(pauseAfter: 100 * 1024, resume: resume))

        let first = try download(slow)
        waitUntil("the slow download writes its partial file") { (size(slow, ".tmp") ?? 0) >= 100 * 1024 }
        let waiting = try download(slow)
        StubServer.enqueue(whole())

        let path = try resolved(try download(other), within: 5)

        XCTAssertEqual(FileManager.default.contents(atPath: path), payload)
        XCTAssertNil(size(slow), "The slow download is still running")
        resume.signal()
        XCTAssertEqual(try resolved(first), try resolved(waiting))
        XCTAssertEqual(StubServer.requests.count, 2)
    }

    // MARK: - sha256

    func testASha256MismatchLeavesNothingBehind() throws {
        let name = fileName("bad")
        StubServer.enqueue(whole())
        let wrong = String(repeating: "0", count: 64)

        guard case .failure(let error) = settle(try download(name, sha256: wrong)) else {
            return XCTFail("A sha256 mismatch must reject")
        }

        XCTAssertTrue(error.localizedDescription.contains("sha256 mismatch: expected \(wrong), got \(sha256(payload))"),
                      error.localizedDescription)
        XCTAssertNil(size(name))
        XCTAssertNil(size(name, ".tmp"))
        XCTAssertNil(size(name, ".tmp.json"))
        XCTAssertFalse(try store.isCached(fileName: name))
    }

    func testAContinuedDownloadIsHashedFromItsFirstByte() throws {
        let name = fileName("hashed")
        try leavePartial(name, cut: 100 * 1024, sha256: sha256(payload))
        StubServer.enqueue(rest(from: 100 * 1024))

        let path = try resolved(try download(name, sha256: sha256(payload)))

        XCTAssertEqual(FileManager.default.contents(atPath: path), payload)
    }
}

/// Where a callback on another thread leaves a promise's result.
private final class ResultBox {
    var result: Result<String, Error>?
}

/// The progress values a download reported, in order (they come from the session's queue).
private final class ProgressLog {
    private let lock = NSLock()
    private var recorded: [Double] = []

    func add(_ value: Double) {
        lock.lock()
        recorded.append(value)
        lock.unlock()
    }

    var values: [Double] {
        lock.lock()
        defer { lock.unlock() }
        return recorded
    }
}

/// One answer of the stub server.
private struct StubAnswer {
    var status = 200
    var headers: [String: String] = [:]
    var body = Data()
    /// After this many bytes of `body` the server waits for `resume` (a slow server).
    var pauseAfter: Int?
    var resume: DispatchSemaphore?
    /// After this many bytes of `body` the connection drops with a network error.
    var dropAfter: Int?
}

/// The stub server: answers each request with the next queued answer (599
/// when none is left) and keeps the requests.
private final class StubServer: URLProtocol {
    private static let lock = NSLock()
    private static var answers: [StubAnswer] = []
    private static var received: [URLRequest] = []

    static func reset() {
        lock.lock()
        defer { lock.unlock() }
        answers = []
        received = []
    }

    static func enqueue(_ answer: StubAnswer) {
        lock.lock()
        defer { lock.unlock() }
        answers.append(answer)
    }

    static func forgetRequests() {
        lock.lock()
        defer { lock.unlock() }
        received = []
    }

    static var requests: [URLRequest] {
        lock.lock()
        defer { lock.unlock() }
        return received
    }

    private var clientRunLoop: CFRunLoop!
    private var clientModes: [CFString] = []
    private var stopped = false

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        // The client expects its calls on this thread, in this run loop's mode
        clientRunLoop = CFRunLoopGetCurrent()
        clientModes = [CFRunLoopMode.defaultMode.rawValue]
        if let mode = CFRunLoopCopyCurrentMode(clientRunLoop), mode != CFRunLoopMode.defaultMode {
            clientModes.append(mode.rawValue)
        }
        Self.lock.lock()
        Self.received.append(request)
        let answer = Self.answers.isEmpty ? StubAnswer(status: 599) : Self.answers.removeFirst()
        Self.lock.unlock()
        NSLog("[StubServer] %@ Range=%@ -> %ld with %ld bytes%@", request.url?.lastPathComponent ?? "?",
              request.value(forHTTPHeaderField: "Range") ?? "-", answer.status, answer.body.count,
              answer.dropAfter.map { ", dropped after \($0)" } ?? "")
        // Sent from another thread, as from the network: a pause in it holds up nothing else
        DispatchQueue.global().async { self.send(answer) }
    }

    override func stopLoading() {
        stopped = true
    }

    private func onClientThread(_ block: @escaping () -> Void) {
        CFRunLoopPerformBlock(clientRunLoop, clientModes as CFArray) {
            if !self.stopped { block() }
        }
        CFRunLoopWakeUp(clientRunLoop)
    }

    private func send(_ answer: StubAnswer) {
        let response = HTTPURLResponse(url: request.url!, statusCode: answer.status, httpVersion: "HTTP/1.1",
                                       headerFields: answer.headers)!
        onClientThread { self.client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed) }
        let end = min(answer.dropAfter ?? answer.body.count, answer.body.count)
        let pause = min(answer.pauseAfter ?? end, end)
        sendBody(answer.body, from: 0, to: pause)
        if answer.pauseAfter != nil {
            answer.resume?.wait()
        }
        sendBody(answer.body, from: pause, to: end)
        if answer.dropAfter != nil {
            onClientThread { self.client?.urlProtocol(self, didFailWithError: URLError(.networkConnectionLost)) }
        } else {
            onClientThread { self.client?.urlProtocolDidFinishLoading(self) }
        }
    }

    private func sendBody(_ body: Data, from start: Int, to end: Int) {
        var at = start
        while at < end {
            let chunk = body.subdata(in: at..<min(end, at + 64 * 1024))
            onClientThread { self.client?.urlProtocol(self, didLoad: chunk) }
            at += chunk.count
        }
    }
}
