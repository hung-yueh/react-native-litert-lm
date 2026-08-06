import XCTest
import UIKit
import NitroModules
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
                XCTAssertEqual(nsError.domain, "LiteRTLM.ModelStore")
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
            XCTAssertEqual(nsError.domain, "LiteRTLM.ModelStore")
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

    func testSendMessageAsyncRejectsWithoutModel() async throws {
        do {
            let promise = try bridge.sendMessageAsync(message: "hello") { _, _ in }
            _ = try await promise.await()
            XCTFail("Should have failed without model")
        } catch {
            let nsError = error as NSError
            XCTAssertEqual(nsError.domain, "LiteRTLM")
            XCTAssertEqual(nsError.code, 400)
        }
    }

    func testSendMessageWithImageAsyncRejectsWithoutModel() async throws {
        // Use a real file so the media validation passes and the
        // no-model guard is what rejects.
        let imagePath = NSTemporaryDirectory() + "no_model_test_image.jpg"
        FileManager.default.createFile(atPath: imagePath, contents: Data([0xFF, 0xD8, 0xFF]))
        defer { try? FileManager.default.removeItem(atPath: imagePath) }
        do {
            let promise = try bridge.sendMessageWithImageAsync(message: "hello", imagePath: imagePath) { _, _ in }
            _ = try await promise.await()
            XCTFail("Should have failed without model")
        } catch {
            let nsError = error as NSError
            XCTAssertEqual(nsError.domain, "LiteRTLM")
            XCTAssertEqual(nsError.code, 400)
        }
    }

    func testSendMessageWithAudioAsyncRejectsWithoutModel() async throws {
        let audioPath = NSTemporaryDirectory() + "no_model_test_audio.wav"
        FileManager.default.createFile(atPath: audioPath, contents: Data([0x52, 0x49, 0x46, 0x46]))
        defer { try? FileManager.default.removeItem(atPath: audioPath) }
        do {
            let promise = try bridge.sendMessageWithAudioAsync(message: "hello", audioPath: audioPath) { _, _ in }
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

    func testDeleteModelCleanupLogic() async throws {
        bridge.loadedModelPath = "/path/to/my_loaded_model.litertlm"

        let promise1 = try bridge.deleteModel(fileName: "other_model.litertlm")
        _ = try await promise1.await()
        XCTAssertEqual(bridge.loadedModelPath, "/path/to/my_loaded_model.litertlm")

        let promise2 = try bridge.deleteModel(fileName: "my_loaded_model.litertlm")
        _ = try await promise2.await()
        XCTAssertNil(bridge.loadedModelPath)
    }

    func testExecutePathPrecedenceOverBuffer() async throws {
        let pathPart = MultimodalPart(
            type: .image,
            text: nil,
            path: "/nonexistent/precedence_test_image.jpg",
            imageBuffer: nil,
            audioBuffer: nil
        )
        
        do {
            let promise = try bridge.execute(parts: [pathPart], onToken: nil, options: nil)
            _ = try await promise.await()
            XCTFail("Should have failed")
        } catch {
            let nsError = error as NSError
            XCTAssertTrue(nsError.localizedDescription.contains("file not found: /nonexistent/precedence_test_image.jpg"))
        }
    }

    func testExecuteTempFileCleanupOnError() async throws {
        let dummyData = Data([0, 1, 2, 3])
        let buffer = try ArrayBuffer.copy(data: dummyData)
        
        let bufferPart = MultimodalPart(
            type: .image,
            text: nil,
            path: nil,
            imageBuffer: buffer,
            audioBuffer: nil
        )
        
        let invalidPathPart = MultimodalPart(
            type: .image,
            text: nil,
            path: "/nonexistent/invalid_file_cleanup_test.jpg",
            imageBuffer: nil,
            audioBuffer: nil
        )
        
        do {
            let promise = try bridge.execute(parts: [bufferPart, invalidPathPart], onToken: nil, options: nil)
            _ = try await promise.await()
            XCTFail("Should have failed")
        } catch {
            let tempDirAfter = try FileManager.default.contentsOfDirectory(atPath: NSTemporaryDirectory())
            let leakedFiles = tempDirAfter.filter { $0.contains("litert_buf_") }
            XCTAssertEqual(leakedFiles.count, 0)
        }
    }

    // MARK: - Image scaling

    /// Writes a JPEG of the given size to a temp path.
    private func makeTestImage(width: Int, height: Int) throws -> String {
        UIGraphicsBeginImageContext(CGSize(width: width, height: height))
        UIColor.red.setFill()
        UIRectFill(CGRect(x: 0, y: 0, width: width, height: height))
        let image = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        let path = NSTemporaryDirectory() + "scale_test_\(width)x\(height).jpg"
        try image.jpegData(compressionQuality: 0.9)!.write(to: URL(fileURLWithPath: path))
        return path
    }

    private func imageSize(atPath path: String) -> CGSize? {
        UIImage(contentsOfFile: path)?.size
    }

    func testScaleImageDownscalesLargeImages() throws {
        let path = try makeTestImage(width: 2048, height: 1024)
        defer { try? FileManager.default.removeItem(atPath: path) }

        let scaledPath = bridge.scaleImageIfNeeded(path)
        defer { if scaledPath != path { try? FileManager.default.removeItem(atPath: scaledPath) } }

        XCTAssertNotEqual(scaledPath, path, "A 2048px image should be rescaled")
        let size = try XCTUnwrap(imageSize(atPath: scaledPath))
        XCTAssertLessThanOrEqual(max(size.width, size.height), 1024)
        // Aspect ratio preserved (2:1)
        XCTAssertEqual(size.width / size.height, 2.0, accuracy: 0.05)
    }

    func testScaleImageLeavesSmallImagesUntouched() throws {
        let path = try makeTestImage(width: 512, height: 256)
        defer { try? FileManager.default.removeItem(atPath: path) }

        let scaledPath = bridge.scaleImageIfNeeded(path)
        XCTAssertEqual(scaledPath, path, "Images within 1024px must pass through unscaled")
    }

    // MARK: - Memory pressure wiring

    func testMemoryWarningCallbackSetAndClearAreIdempotent() throws {
        var fired = 0
        try bridge.setMemoryWarningCallback { _, _ in fired += 1 }
        try bridge.setMemoryWarningCallback { _, _ in fired += 1 } // replace, not stack
        try bridge.clearMemoryWarningCallback()
        try bridge.clearMemoryWarningCallback() // double-clear must not throw
        // Registration alone must not fire spuriously.
        XCTAssertEqual(fired, 0)
    }

    // MARK: - Lifecycle & concurrency

    func testInstanceIsReusableAfterUnload() async throws {
        _ = try bridge.unload() // unload with nothing loaded is a no-op

        // First load attempt fails (nonexistent file) …
        do {
            _ = try await bridge.loadModel(modelPath: "/nonexistent/a.litertlm", config: nil).await()
            XCTFail("Load of a nonexistent model should reject")
        } catch { /* expected */ }

        _ = try bridge.unload()

        // … and the same instance accepts another load attempt afterwards.
        do {
            _ = try await bridge.loadModel(modelPath: "/nonexistent/b.litertlm", config: nil).await()
            XCTFail("Load of a nonexistent model should reject")
        } catch {
            XCTAssertFalse(try bridge.isReady())
        }
    }

    func testConcurrentExecuteAndUnloadDoesNotCrash() async throws {
        // Hammer the serial engine queue from two directions. Every call is
        // expected to reject (no model); the invariant is no crash/deadlock.
        await withTaskGroup(of: Void.self) { group in
            group.addTask {
                for _ in 0..<20 {
                    if let promise = try? self.bridge.execute(
                        parts: [MultimodalPart(type: .text, text: "hi", path: nil, imageBuffer: nil, audioBuffer: nil)],
                        onToken: nil,
                        options: nil
                    ) {
                        _ = try? await promise.await()
                    }
                }
            }
            group.addTask {
                for _ in 0..<20 {
                    _ = try? self.bridge.unload()
                    try? self.bridge.resetConversation(historyJson: nil, systemPrompt: nil)
                }
            }
        }
        XCTAssertFalse(try bridge.isReady())
    }

    // MARK: - Opt-in real inference smoke test
    //
    // Set LITERTLM_TEST_MODEL to the absolute path of a local .litertlm file
    // (e.g. Gemma 3 1B) to run a true end-to-end generation:
    //   LITERTLM_TEST_MODEL=$HOME/models/gemma3-1b.litertlm xcodebuild test …
    // Skipped by default so CI stays device- and download-free.

    func testRealInferenceSmoke() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment["LITERTLM_TEST_MODEL"],
              FileManager.default.fileExists(atPath: modelPath) else {
            throw XCTSkip("LITERTLM_TEST_MODEL not set — skipping real inference smoke test")
        }

        _ = try await bridge.loadModel(modelPath: modelPath, config: nil).await()
        XCTAssertTrue(try bridge.isReady())

        var streamed = ""
        let promise = try bridge.execute(
            parts: [MultimodalPart(type: .text, text: "Reply with one short sentence.", path: nil, imageBuffer: nil, audioBuffer: nil)],
            onToken: { token, _ in streamed += token },
            options: nil
        )
        let full = try await promise.await()

        XCTAssertFalse(full.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        XCTAssertFalse(streamed.isEmpty, "Streaming callback should have received tokens")
        let stats = try bridge.getStats()
        XCTAssertGreaterThan(stats.totalTokens, 0)

        _ = try bridge.unload()
        XCTAssertFalse(try bridge.isReady())
    }
}
