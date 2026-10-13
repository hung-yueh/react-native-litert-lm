import XCTest
import NitroModules
import CLiteRTLM
@testable import LiteRTLM

/// Real on-device embeddings in the simulator (LiteRT-LM EmbeddingEngine).
///
/// Skipped unless an EmbeddingGemma 2 bundle path is provided:
///
///   xcodebuild test … \
///     TEST_RUNNER_LITERTLM_TEST_EMBEDDER=$HOME/.litert-models/embeddinggemma-2-text-270m.litertlm \
///     TEST_RUNNER_LITERTLM_TEST_EMBEDDER_MM=$HOME/.litert-models/embeddinggemma-2-740m.litertlm
///
/// The text checks are semantic sanity, not benchmarks: a query must land
/// nearer its matching passage than an unrelated one.
final class HybridEmbedderIntegrationTests: XCTestCase {

    static func env(_ key: String) -> String? {
        guard let path = ProcessInfo.processInfo.environment[key],
              FileManager.default.fileExists(atPath: path) else { return nil }
        return path
    }

    private func text(_ s: String) -> MultimodalPart {
        MultimodalPart(type: .text, text: s, path: nil, imageBuffer: nil, audioBuffer: nil)
    }

    private func file(_ type: PartType, _ path: String) -> MultimodalPart {
        MultimodalPart(type: type, text: nil, path: path, imageBuffer: nil, audioBuffer: nil)
    }

    private func options(normalize: Bool = true, outputSize: Double? = nil) -> EmbeddingOptions {
        EmbeddingOptions(normalize: normalize, outputSize: outputSize, insertSpecialTokens: nil,
                         visionTokensPerImage: nil)
    }

    private func floats(_ buffer: ArrayBuffer) -> [Float] {
        let data = buffer.toData(copyIfNeeded: true)
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    private func dot(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
    }

    private func loaded(_ path: String, _ config: EmbedderConfig? = nil) async throws -> HybridEmbedder {
        let embedder = HybridEmbedder()
        _ = try await embedder.loadModel(modelPath: path, config: config).await()
        XCTAssertTrue(try embedder.isReady())
        return embedder
    }

    func testE1_TextEmbeddingsAreSemanticallySound() async throws {
        guard let path = Self.env("LITERTLM_TEST_EMBEDDER") else {
            throw XCTSkip("Set TEST_RUNNER_LITERTLM_TEST_EMBEDDER to an EmbeddingGemma 2 bundle")
        }
        let info = try await HybridModelInspector().inspect(modelPath: path).await()
        print("[embed] bundle: type=\(info.modelType.stringValue) dim=\(Int(info.embeddingDimension)) " +
              "inputs=\(info.embeddingSignatureSelection.map { Int($0) })")
        XCTAssertEqual(info.modelType, .embedding)

        let t0 = Date()
        let embedder = try await loaded(path, EmbedderConfig(backend: nil, visionBackend: nil, audioBackend: nil,
                                                              maxInputLength: 256, visionTokensPerImage: nil,
                                                              activationDataType: nil, forceLoad: nil))
        defer { try? embedder.close() }
        print(String(format: "[embed] loaded in %.2fs", Date().timeIntervalSince(t0)))

        func embed(_ s: String, _ opts: EmbeddingOptions? = nil) async throws -> [Float] {
            floats(try await embedder.embed(parts: [text(s)], options: opts ?? options()).await())
        }
        let started = Date()
        let query = try await embed("task: search result | query: How do I bake bread at home?")
        let latency = Date().timeIntervalSince(started)
        let match = try await embed("title: none | text: A simple recipe for a crusty sourdough loaf.")
        let other = try await embed("title: none | text: The stock market fell sharply on Tuesday.")

        XCTAssertEqual(query.count, Int(info.embeddingDimension) > 0 ? Int(info.embeddingDimension) : 768)
        XCTAssertEqual(dot(query, query), 1, accuracy: 1e-3, "normalize: true must give unit vectors")
        let near = dot(query, match), far = dot(query, other)
        print(String(format: "[embed] 1 vector in %.0f ms; similarity: bread↔recipe %.3f, bread↔stocks %.3f",
                     latency * 1000, near, far))
        XCTAssertGreaterThan(near, far, "the query should land nearer the matching passage")

        // Matryoshka truncation
        let short = try await embed("task: search result | query: How do I bake bread at home?",
                                    options(outputSize: 256))
        XCTAssertEqual(short.count, 256)

        // A batch equals the same inputs one by one
        let batch = try await embedder.embedBatch(
            inputs: [[text("title: none | text: A simple recipe for a crusty sourdough loaf.")],
                     [text("title: none | text: The stock market fell sharply on Tuesday.")]],
            options: options()
        ).await().map(floats)
        XCTAssertEqual(batch.count, 2)
        XCTAssertEqual(dot(batch[0], match), 1, accuracy: 1e-3)
        XCTAssertEqual(dot(batch[1], other), 1, accuracy: 1e-3)
    }

    func testE2_EmbedWithoutALoadedModelIsTyped() async throws {
        do {
            _ = try await HybridEmbedder().embed(parts: [text("x")], options: nil).await()
            XCTFail("embedding without a model must fail")
        } catch let e as LiteRTLMNativeError {
            XCTAssertEqual(e.code, "MODEL_NOT_LOADED")
            XCTAssertEqual(e.phase, .embed)
        }
    }

    /// close() never waits for engine work: readiness drops at once, the
    /// in-flight batch still completes, and the instance loads again in call
    /// order. Media files are read on the queue, so a bad part rejects (typed).
    func testE4_CloseDoesNotWaitForEngineWork() async throws {
        guard let path = Self.env("LITERTLM_TEST_EMBEDDER") else {
            throw XCTSkip("Set TEST_RUNNER_LITERTLM_TEST_EMBEDDER to an EmbeddingGemma 2 bundle")
        }
        let embedder = try await loaded(path)
        defer { try? embedder.close() }

        let notes = (0..<32).map { [text("title: none | text: Note \($0): water the tomato plants in July.")] }
        let batch = try embedder.embedBatch(inputs: notes, options: options())
        let started = Date()
        try embedder.close()
        let closeMs = Date().timeIntervalSince(started) * 1000
        print(String(format: "[embed] close() returned in %.2f ms during a 32-input batch", closeMs))
        XCTAssertLessThan(closeMs, 50, "close() must not wait for the batch")
        XCTAssertFalse(try embedder.isReady())
        let vectors = try await batch.await()
        XCTAssertEqual(vectors.count, 32)

        _ = try await embedder.loadModel(modelPath: path, config: nil).await()
        XCTAssertTrue(try embedder.isReady())

        for part in [file(.image, "/nope.jpg"),
                     MultimodalPart(type: .audio, text: nil, path: nil, imageBuffer: nil, audioBuffer: nil)] {
            do {
                _ = try await embedder.embed(parts: [part], options: nil).await()
                XCTFail("a bad media part must reject")
            } catch let e as LiteRTLMNativeError {
                XCTAssertEqual(e.code, "INVALID_INPUT")
            }
        }
    }

    /// Text, image and audio share one space: each caption must land nearer
    /// its own media than the other caption does.
    func testE3_CrossModalEmbeddings() async throws {
        guard let path = Self.env("LITERTLM_TEST_EMBEDDER_MM") else {
            throw XCTSkip("Set TEST_RUNNER_LITERTLM_TEST_EMBEDDER_MM to embeddinggemma-2-740m")
        }
        let embedder = try await loaded(path)
        defer { try? embedder.close() }
        let image = HybridLiteRTLMIntegrationTests.fixture("test.jpeg")
        let audio = HybridLiteRTLMIntegrationTests.fixture("test.wav")

        func embed(_ parts: [MultimodalPart]) async throws -> [Float] {
            floats(try await embedder.embed(parts: parts, options: options()).await())
        }
        var started = Date()
        let picture = try await embed([file(.image, image)])
        let imageMs = Date().timeIntervalSince(started) * 1000
        started = Date()
        let clip = try await embed([file(.audio, audio)])
        let audioMs = Date().timeIntervalSince(started) * 1000
        let cake = try await embed([text("task: search result | query: a person holding a birthday cake")])
        let mountain = try await embed([text("task: search result | query: a snowy mountain landscape")])
        let greeting = try await embed([text("task: search result | query: people greeting each other")])
        let engine = try await embed([text("task: search result | query: a car engine revving")])

        print(String(format: "[embed][mm] image %.0f ms, audio %.0f ms; image↔cake %.3f vs mountain %.3f; " +
                     "audio↔greeting %.3f vs engine %.3f",
                     imageMs, audioMs, dot(picture, cake), dot(picture, mountain),
                     dot(clip, greeting), dot(clip, engine)))
        XCTAssertGreaterThan(dot(picture, cake), dot(picture, mountain))
        XCTAssertGreaterThan(dot(clip, greeting), dot(clip, engine))
    }
}
