//
//  HybridLiteRTLM+Tools.swift
//  react-native-litert-lm
//
//  The tool-calling round trip: a message of tool results in, structured
//  tool calls out of the transcript, and the engine's message format for
//  replaying a transcript into a new conversation.
//

import Foundation
import NitroModules

extension HybridLiteRTLM {

    /// `{"role":"tool","content":[{"type":"tool_response","name":…,"response":…}]}`.
    /// Each result is validated as JSON, then spliced in verbatim so its key
    /// order and number formats reach the model unchanged.
    static func toolMessageJson(_ results: [(name: String, json: String)]) throws -> String {
        var items: [String] = []
        for result in results {
            guard jsonValue(result.json) != nil else {
                throw LiteRTLMNativeError(.execute, status: "INVALID_ARGUMENT", code: "INVALID_INPUT",
                                          "The result for tool '\(result.name)' is not valid JSON")
            }
            let name = try JSONSerialization.data(withJSONObject: result.name, options: [.fragmentsAllowed])
            items.append("{\"type\":\"tool_response\",\"name\":\(String(decoding: name, as: UTF8.self))," +
                         "\"response\":\(result.json)}")
        }
        return "{\"role\":\"tool\",\"content\":[\(items.joined(separator: ","))]}"
    }

    /// Splits `<tool_call>{"name":…,"arguments":{…}}</tool_call>` markers out
    /// of a response: the remaining text, and the calls. A marker whose JSON
    /// doesn't parse stays in the text.
    static func splitToolCalls(_ response: String) -> (text: String, calls: [ToolCall]) {
        guard response.contains("<tool_call>") else { return (response, []) }
        var calls: [ToolCall] = []
        var text = ""
        var rest = Substring(response)
        while let start = rest.range(of: "<tool_call>"),
              let end = rest.range(of: "</tool_call>", range: start.upperBound..<rest.endIndex) {
            text += rest[..<start.lowerBound]
            if let call = parseToolCall(String(rest[start.upperBound..<end.lowerBound])) {
                calls.append(call)
            } else {
                text += rest[start.lowerBound..<end.upperBound]
            }
            rest = rest[end.upperBound...]
        }
        text += rest
        return (text.trimmingCharacters(in: .whitespacesAndNewlines), calls)
    }

    private static func parseToolCall(_ body: String) -> ToolCall? {
        guard let object = jsonValue(body) as? [String: Any],
              let name = object["name"] as? String,
              let arguments = try? JSONSerialization.data(
                withJSONObject: object["arguments"] ?? [String: Any](), options: [.fragmentsAllowed])
        else { return nil }
        return ToolCall(name: name, argumentsJson: String(decoding: arguments, as: UTF8.self))
    }

    /// A transcript from JS (the JSON of `getHistory()`) as Messages. An
    /// unknown role reads as a user turn; "assistant" reads as the model.
    static func parseTranscript(_ json: String?) -> [Message] {
        guard let json = json,
              let items = jsonValue(json) as? [[String: Any]] else { return [] }
        return items.compactMap { item in
            guard let roleName = item["role"] as? String,
                  let content = item["content"] as? String else { return nil }
            let role = roleName == "assistant" ? Role.model : (Role(fromString: roleName) ?? .user)
            let calls = (item["toolCalls"] as? [[String: Any]])?.compactMap { call -> ToolCall? in
                guard let name = call["name"] as? String,
                      let arguments = call["argumentsJson"] as? String else { return nil }
                return ToolCall(name: name, argumentsJson: arguments)
            } ?? []
            return Message(role: role, content: content, toolCalls: calls.isEmpty ? nil : calls,
                           toolName: item["toolName"] as? String)
        }
    }

    /// The engine's message JSON for replaying a transcript: model turns as
    /// "assistant" with their tool_calls, tool turns as tool_response content —
    /// the shapes chat templates render natively.
    static func engineMessagesJson(_ messages: [Message]) -> String? {
        guard !messages.isEmpty else { return nil }
        let items: [[String: Any]] = messages.map { message in
            switch message.role {
            case .tool:
                return ["role": "tool", "content": [[
                    "type": "tool_response",
                    "name": message.toolName ?? "",
                    "response": jsonValue(message.content) ?? message.content,
                ]]]
            case .model:
                var item: [String: Any] = ["role": "assistant", "content": message.content]
                if let calls = message.toolCalls, !calls.isEmpty {
                    item["tool_calls"] = calls.map { call -> [String: Any] in
                        ["type": "function",
                         "function": ["name": call.name,
                                      "arguments": jsonValue(call.argumentsJson) ?? [String: Any]()]]
                    }
                }
                return item
            default:
                return ["role": message.role.stringValue, "content": message.content]
            }
        }
        guard let data = try? JSONSerialization.data(withJSONObject: items) else { return nil }
        return String(decoding: data, as: UTF8.self)
    }

    private static func jsonValue(_ json: String) -> Any? {
        guard let data = json.data(using: .utf8) else { return nil }
        return try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
    }
}
