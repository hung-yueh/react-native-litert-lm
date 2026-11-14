//
//  HybridLiteRTLM+Streaming.swift
//  react-native-litert-lm
//
//  Streaming context and callback runners for the unified execute pipeline.
//

import Foundation
import NitroModules
import CLiteRTLM

class ExecuteStreamContext {
    let userLabel: String
    let inputTurns: [Message]
    /// The conversation this turn belongs to (see Snapshot.epoch).
    let epoch: Int
    /// The conversation it streams in. Not freed before streamEnded() (see settleStreams()).
    let conversation: OpaquePointer
    /// With LLMConfig.exactStats: the engine's turn counts before the send.
    let engineMark: HybridLiteRTLM.EngineTurnMark?
    let startTime: Date
    let onToken: (_ token: String, _ done: Bool) -> Void
    let promise: Promise<String>
    let parent: HybridLiteRTLM
    let cleanup: () -> Void
    var rawResponse: String = ""
    var fullResponse: String = ""
    var lastEmittedLength: Int = 0
    /// When the first token reached the callback.
    var firstTokenTime: Date?
    /// A chunk accessor that failed (C API 1.0 accessors return a status).
    /// The stream still runs to its last chunk, which then fails with this.
    var readFailure: LiteRTLMNativeError?

    init(
        userLabel: String,
        inputTurns: [Message],
        epoch: Int,
        conversation: OpaquePointer,
        engineMark: HybridLiteRTLM.EngineTurnMark?,
        startTime: Date,
        onToken: @escaping (_ token: String, _ done: Bool) -> Void,
        promise: Promise<String>,
        parent: HybridLiteRTLM,
        cleanup: @escaping () -> Void
    ) {
        self.userLabel = userLabel
        self.inputTurns = inputTurns
        self.epoch = epoch
        self.conversation = conversation
        self.engineMark = engineMark
        self.startTime = startTime
        self.onToken = onToken
        self.promise = promise
        self.parent = parent
        self.cleanup = cleanup
    }
}

extension HybridLiteRTLM {

    func runExecuteStreaming(
        conversation: OpaquePointer,
        msgJson: String,
        userLabel: String,
        inputTurns: [Message],
        epoch: Int,
        onToken: @escaping (_ token: String, _ done: Bool) -> Void,
        promise: Promise<String>,
        cleanup: @escaping () -> Void,
        options: ExecuteOptions? = nil
    ) {
        // Per-message optional args must outlive the async stream — freed in
        // the context's cleanup, which runs exactly once on completion/error.
        let optionalArgs: OpaquePointer?
        do { optionalArgs = try makeOptionalArgs(options) } catch {
            cleanup()
            promise.reject(withError: error)
            return
        }
        let fullCleanup = {
            cleanup()
            if let optionalArgs = optionalArgs {
                litert_lm_conversation_optional_args_delete(optionalArgs)
            }
        }
        // exactStats: the engine's turns so far; this message's are the next ones.
        let mark = exactStats ? Self.engineTurnMark(conversation) : nil
        let ctx = ExecuteStreamContext(
            userLabel: userLabel, inputTurns: inputTurns, epoch: epoch,
            conversation: conversation, engineMark: mark, startTime: Date(),
            onToken: onToken, promise: promise, parent: self,
            cleanup: fullCleanup
        )
        let ptr = Unmanaged.passRetained(ctx).toOpaque()

        // v0.15: the callback receives an opaque LiteRtLmStreamChunk; text,
        // finality, and errors are read through accessors. The chunk (and any
        // string it returns) is only valid for the duration of the call. It
        // runs on the engine's thread and must not call into the engine or
        // this conversation (c/engine.h "Threading model"): work that does
        // moves to `queue`.
        let cb: LiteRtLmStreamCallback = { ptr, chunk in
            guard let ptr = ptr, let chunk = chunk else { return }
            let ctx = Unmanaged<ExecuteStreamContext>.fromOpaque(ptr).takeUnretainedValue()
            ctx.parent.handleExecuteStreamChunk(chunk, ctx: ctx, streamPtr: ptr)
        }

        streamStarted()
        do {
            try LiteRTLMNativeError.check(.execute, "Failed to start streaming") {
                litert_lm_conversation_send_message_stream(conversation, msgJson, nil, optionalArgs, cb, ptr)
            }
        } catch {
            Unmanaged<ExecuteStreamContext>.fromOpaque(ptr).release()
            fullCleanup()
            streamEnded()
            promise.reject(withError: error)
            return
        }

        // singleThreadedExecution: queued work runs only while a caller drives
        // it (c/conversation.h, LiteRT-LM 0.19), so drive this stream here, on
        // `queue` — never the JS thread. Its callbacks run inside this call.
        if singleThreadedExecution == true {
            do {
                try LiteRTLMNativeError.check(.execute, "Failed to run the stream (singleThreadedExecution)") {
                    litert_lm_conversation_wait_until_done(conversation)
                }
            } catch {
                // The stream reports its own failure through its last chunk.
                NSLog("[LiteRTLM] %@", "\(error)")
            }
        }
    }

    /// One chunk, on the engine's thread (or inside wait_until_done on `queue`).
    func handleExecuteStreamChunk(_ chunk: OpaquePointer, ctx: ExecuteStreamContext, streamPtr: UnsafeMutableRawPointer) {
        // C API 1.0: every accessor returns a status. They fail only for a
        // NULL argument, which never reaches here; if one ever does, the
        // stream still ends at its last chunk, never earlier: freeing the
        // context sooner would free it under the callbacks still to come.
        func read<T>(_ what: String, _ initial: T, _ call: (UnsafeMutablePointer<T>) -> LiteRtLmStatusCode) -> T {
            do {
                return try LiteRTLMNativeError.read(.execute, "Failed to read a stream chunk's \(what)", initial, call)
            } catch {
                if ctx.readFailure == nil { ctx.readFailure = error as? LiteRTLMNativeError }
                return initial
            }
        }

        if let errorMsg = read("error", nil, { litert_lm_stream_chunk_get_error(chunk, $0) }) {
            // Stream errors arrive as text on the engine's thread — classify
            // from the absl::Status prefix ("RESOURCE_EXHAUSTED: …") if any.
            let msg = String(cString: errorMsg)
            failExecuteStream(ctx: ctx, streamPtr: streamPtr,
                              error: LiteRTLMNativeError.fromStatusMessage(.execute, msg), text: msg)
            return
        }

        if read("finality", false, { litert_lm_stream_chunk_is_final(chunk, $0) }) {
            if let failure = ctx.readFailure {
                failExecuteStream(ctx: ctx, streamPtr: streamPtr, error: failure, text: failure.message)
            } else {
                finalizeExecuteStream(ctx: ctx, streamPtr: streamPtr)
            }
            return
        }

        if let text = read("text", nil, { litert_lm_stream_chunk_get_text(chunk, $0) }) {
            emitExecuteStreamChunk(ctx: ctx, chunk: text)
        }
    }

    /// End a stream with an error: its last callback.
    private func failExecuteStream(
        ctx: ExecuteStreamContext, streamPtr: UnsafeMutableRawPointer, error: LiteRTLMNativeError, text: String
    ) {
        // A cancel (close/unload/reset mid-stream) is a deliberate stop,
        // not a failure to show: end the stream without error text. The
        // promise still rejects with CANCELLED.
        ctx.onToken(error.status == "CANCELLED" ? "" : "Error: \(text)", true)
        ctx.promise.reject(withError: error)
        // The message options are the engine's: free them on `queue`.
        queue.async { ctx.cleanup() }
        Unmanaged<ExecuteStreamContext>.fromOpaque(streamPtr).release()
        streamEnded()
    }

    func finalizeExecuteStream(ctx: ExecuteStreamContext, streamPtr: UnsafeMutableRawPointer) {
        let cleaned = stripControlTokens(ctx.rawResponse)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        var finalText = cleaned
        if !ctx.userLabel.isEmpty && finalText.hasPrefix(ctx.userLabel) {
            finalText = String(finalText.dropFirst(ctx.userLabel.count))
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if finalText.count > ctx.lastEmittedLength {
            let si = finalText.index(finalText.startIndex, offsetBy: ctx.lastEmittedLength)
            if ctx.firstTokenTime == nil { ctx.firstTokenTime = Date() }
            ctx.onToken(String(finalText[si...]), false)
        }
        ctx.fullResponse = finalText

        let endTime = Date()
        // The engine is done with the conversation (see settleStreams()).
        streamEnded()
        // The engine's numbers (exactStats) are read on `queue`, never here:
        // this callback must not call into the conversation. A message sent
        // during this stream can commit first; the mark keeps this message's
        // own turns apart from its.
        queue.async {
            self.commitExecuteTurn(
                epoch: ctx.epoch,
                inputTurns: ctx.inputTurns,
                modelResponse: ctx.fullResponse,
                startTime: ctx.startTime,
                endTime: endTime,
                // No text reached the callback (all of the budget went to
                // thinking, say): no first token, so 0, as on Android.
                firstTokenTime: ctx.firstTokenTime ?? ctx.startTime,
                engineMark: ctx.engineMark,
                conversation: ctx.conversation
            )
            ctx.onToken("", true)
            ctx.cleanup()
            ctx.promise.resolve(withResult: ctx.fullResponse)
            Unmanaged<ExecuteStreamContext>.fromOpaque(streamPtr).release()
        }
    }

    func emitExecuteStreamChunk(ctx: ExecuteStreamContext, chunk: UnsafePointer<CChar>) {
        let token = String(cString: chunk)
        let raw = token.hasPrefix("{") && token.contains("\"role\"")
            ? extractTextFromResponse(token) : token
        ctx.rawResponse += raw
        let cleaned = stripControlTokens(ctx.rawResponse)
            .trimmingLeadingCharacters(in: .whitespacesAndNewlines)
        var processed = cleaned
        if !ctx.userLabel.isEmpty && processed.hasPrefix(ctx.userLabel) {
            processed = String(processed.dropFirst(ctx.userLabel.count))
                .trimmingLeadingCharacters(in: .whitespacesAndNewlines)
        }
        let safe = safeEmitLength(processed)
        if safe > ctx.lastEmittedLength {
            let chars = Array(processed)
            if ctx.firstTokenTime == nil { ctx.firstTokenTime = Date() }
            ctx.onToken(String(chars[ctx.lastEmittedLength..<safe]), false)
            ctx.lastEmittedLength = safe
        }
    }
}

private extension String {
    func trimmingLeadingCharacters(in characterSet: CharacterSet) -> String {
        guard let index = firstIndex(where: { char in
            !char.unicodeScalars.allSatisfy { characterSet.contains($0) }
        }) else {
            return ""
        }
        return String(self[index...])
    }
}
