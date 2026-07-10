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
    let startTime: Date
    let onToken: (_ token: String, _ done: Bool) -> Void
    let promise: Promise<String>
    let parent: HybridLiteRTLM
    let cleanup: () -> Void
    var rawResponse: String = ""
    var fullResponse: String = ""
    var lastEmittedLength: Int = 0
    var tokenCount: Int = 0

    init(
        userLabel: String,
        startTime: Date,
        onToken: @escaping (_ token: String, _ done: Bool) -> Void,
        promise: Promise<String>,
        parent: HybridLiteRTLM,
        cleanup: @escaping () -> Void
    ) {
        self.userLabel = userLabel
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
        onToken: @escaping (_ token: String, _ done: Bool) -> Void,
        promise: Promise<String>,
        cleanup: @escaping () -> Void,
        options: ExecuteOptions? = nil
    ) {
        // Per-message optional args must outlive the async stream — freed in
        // the context's cleanup, which runs exactly once on completion/error.
        let optionalArgs = makeOptionalArgs(options)
        let fullCleanup = {
            cleanup()
            if let optionalArgs = optionalArgs {
                litert_lm_conversation_optional_args_delete(optionalArgs)
            }
        }
        let ctx = ExecuteStreamContext(
            userLabel: userLabel, startTime: Date(),
            onToken: onToken, promise: promise, parent: self,
            cleanup: fullCleanup
        )
        let ptr = Unmanaged.passRetained(ctx).toOpaque()

        let cb: LiteRtLmStreamCallback = { ptr, chunk, isFinal, errorMsg in
            guard let ptr = ptr else { return }
            let ctx = Unmanaged<ExecuteStreamContext>.fromOpaque(ptr).takeUnretainedValue()

            if let errorMsg = errorMsg {
                let msg = String(cString: errorMsg)
                ctx.onToken("Error: \(msg)", true)
                ctx.cleanup()
                ctx.promise.reject(withError: NSError(domain: "LiteRTLM", code: 500,
                    userInfo: [NSLocalizedDescriptionKey: msg]))
                Unmanaged<ExecuteStreamContext>.fromOpaque(ptr).release()
                return
            }

            if isFinal {
                ctx.parent.finalizeExecuteStream(ctx: ctx, streamPtr: ptr)
                return
            }

            if let chunk = chunk {
                ctx.parent.emitExecuteStreamChunk(ctx: ctx, chunk: chunk)
            }
        }

        let status = litert_lm_conversation_send_message_stream(
            conversation, msgJson, nil, optionalArgs, cb, ptr)
        if status != 0 {
            Unmanaged<ExecuteStreamContext>.fromOpaque(ptr).release()
            fullCleanup()
            promise.reject(withError: NSError(domain: "LiteRTLM", code: Int(status),
                userInfo: [NSLocalizedDescriptionKey: "LiteRTLM: execute streaming failed."]))
        }
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
            ctx.onToken(String(finalText[si...]), false)
        }
        ctx.fullResponse = finalText

        queue.async {
            self.commitExecuteTurn(
                userLabel: ctx.userLabel,
                modelResponse: ctx.fullResponse,
                startTime: ctx.startTime,
                conversation: self.conversation,
                tokenCount: ctx.tokenCount
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
            ctx.onToken(String(chars[ctx.lastEmittedLength..<safe]), false)
            ctx.lastEmittedLength = safe
            ctx.tokenCount += 1
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
