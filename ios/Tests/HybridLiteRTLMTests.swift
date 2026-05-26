import XCTest
@testable import LiteRTLM

class HybridLiteRTLMTests: XCTestCase {
    var bridge: HybridLiteRTLM!

    override func setUp() {
        super.setUp()
        bridge = HybridLiteRTLM()
    }

    override func tearDown() {
        try? bridge.close()
        bridge = nil
        super.tearDown()
    }

    func testPathTraversalRejection() async throws {
        let traversals = ["../../etc/passwd", "/absolute/path/file", "subdir\\..\\file", "..", "../", "..\\"]
        for traversal in traversals {
            do {
                let promise = try bridge.deleteModel(fileName: traversal)
                _ = try await promise.await()
                XCTFail("Should have failed for traversal: \(traversal)")
            } catch {
                let nsError = error as NSError
                XCTAssertEqual(nsError.domain, "LiteRTLM")
                XCTAssertEqual(nsError.code, 400)
                XCTAssertTrue(nsError.localizedDescription.contains("path traversal") || nsError.localizedDescription.contains("directory separators"))
            }
        }
    }

    func testNonHTTPSDownloadRejection() async throws {
        do {
            let promise = try bridge.downloadModel(url: "http://insecure-domain.com/model.bin", fileName: "model.bin", onProgress: nil)
            _ = try await promise.await()
            XCTFail("Should have blocked insecure HTTP downloads")
        } catch {
            let nsError = error as NSError
            XCTAssertEqual(nsError.domain, "LiteRTLM")
            XCTAssertEqual(nsError.code, 400)
            XCTAssertTrue(nsError.localizedDescription.contains("HTTPS is required"))
        }
    }

    func testMemoryTelemetry() {
        XCTAssertNoThrow(try bridge.getMemoryUsage())
        if let mem = try? bridge.getMemoryUsage() {
            XCTAssertGreaterThanOrEqual(mem.nativeHeapBytes, 0.0)
            XCTAssertGreaterThanOrEqual(mem.residentBytes, 0.0)
            XCTAssertGreaterThanOrEqual(mem.availableMemoryBytes, 0.0)
        }
    }

    func testSendMessageWithImageAsyncRejectsWithoutModel() async throws {
        do {
            let promise = try bridge.sendMessageWithImageAsync(message: "hello", imagePath: "/tmp/image.jpg") { _, _ in }
            _ = try await promise.await()
            XCTFail("Should have failed without model")
        } catch {
            let nsError = error as NSError
            XCTAssertEqual(nsError.domain, "LiteRTLM")
            XCTAssertEqual(nsError.code, 400)
        }
    }

    func testSendMessageWithAudioAsyncRejectsWithoutModel() async throws {
        do {
            let promise = try bridge.sendMessageWithAudioAsync(message: "hello", audioPath: "/tmp/audio.wav") { _, _ in }
            _ = try await promise.await()
            XCTFail("Should have failed without model")
        } catch {
            let nsError = error as NSError
            XCTAssertEqual(nsError.domain, "LiteRTLM")
            XCTAssertEqual(nsError.code, 400)
        }
    }

    func testSendMessageWithImageAsyncRejectsFileNotFound() async throws {
        do {
            let promise = try bridge.sendMessageWithImageAsync(message: "hello", imagePath: "/nonexistent/image.jpg") { _, _ in }
            _ = try await promise.await()
            XCTFail("Should have failed without model")
        } catch {
            let nsError = error as NSError
            XCTAssertEqual(nsError.domain, "LiteRTLM")
        }
    }

    func testSendMessageWithAudioAsyncRejectsFileNotFound() async throws {
        do {
            let promise = try bridge.sendMessageWithAudioAsync(message: "hello", audioPath: "/nonexistent/audio.wav") { _, _ in }
            _ = try await promise.await()
            XCTFail("Should have failed without model")
        } catch {
            let nsError = error as NSError
            XCTAssertEqual(nsError.domain, "LiteRTLM")
        }
    }

    func testInitialStats() {
        XCTAssertNoThrow(try bridge.getStats())
        if let stats = try? bridge.getStats() {
            XCTAssertEqual(stats.promptTokens, 0.0)
            XCTAssertEqual(stats.completionTokens, 0.0)
            XCTAssertEqual(stats.totalTokens, 0.0)
            XCTAssertEqual(stats.timeToFirstToken, 0.0)
            XCTAssertEqual(stats.totalTime, 0.0)
            XCTAssertEqual(stats.tokensPerSecond, 0.0)
        }
    }
}
