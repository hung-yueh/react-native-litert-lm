//
//  HybridLiteRTLM+Tools.swift
//  react-native-litert-lm
//
//  The tool-calling round trip: a message of tool results in, structured
//  tool calls out of the transcript, and the engine's message format for
//  replaying a transcript into a new conversation.
//
//  Tool JSON is written and read as text, never through a dictionary: Swift
//  dictionaries change their order with every launch, and chat templates write
//  a tool's fields in the order they get them. (LFM3's template does, and the
//  model named its call after the description whenever that came first.)
//

import Foundation
import NitroModules

extension HybridLiteRTLM {

    /// The engine's tools JSON,
    /// `[{"type":"function","function":{"name":…,"description":…,"parameters":…}}]`,
    /// in that order, with each tool's schema spliced in as the app wrote it.
    static func toolsJson(_ tools: [ToolDefinition]) -> String {
        let items = tools.map { tool -> String in
            var parameters = tool.parametersJson
            if tool.parametersJson.data(using: .utf8)
                .flatMap({ try? JSONSerialization.jsonObject(with: $0, options: []) }) == nil {
                // Don't silently drop the schema — the model would see a
                // tool with no parameters and hallucinate arguments.
                NSLog("[LiteRTLM] Tool '%@' has unparseable parametersJson — sending empty schema", tool.name)
                parameters = "{}"
            }
            return "{\"type\":\"function\",\"function\":{\"name\":\(JSONText.literal(tool.name))," +
                   "\"description\":\(JSONText.literal(tool.description)),\"parameters\":\(parameters)}}"
        }
        return "[\(items.joined(separator: ","))]"
    }

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
            items.append(toolResponseJson(name: result.name, response: result.json))
        }
        return "{\"role\":\"tool\",\"content\":[\(items.joined(separator: ","))]}"
    }

    /// One engine tool call (`{"type":"function","function":{"name":…,"arguments":…}}`,
    /// or the function object itself) as a `<tool_call>{"name":…,"arguments":…}</tool_call>`
    /// marker, made from the call's own text so the arguments keep the model's order.
    static func toolCallMarker(_ call: String) -> String {
        let function = JSONText.member("function", of: call).flatMap { $0.hasPrefix("{") ? $0 : nil } ?? call
        let name = JSONText.member("name", of: function) ?? "\"\""
        let arguments = JSONText.member("arguments", of: function) ?? "{}"
        // `</` only occurs inside JSON strings, where `<\/` reads the same; an
        // argument's `</tool_call>` would otherwise end the marker early.
        let payload = "{\"name\":\(name),\"arguments\":\(arguments)}"
            .replacingOccurrences(of: "</", with: "<\\/")
        return "<tool_call>\(payload)</tool_call>"
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
              let name = object["name"] as? String else { return nil }
        // The arguments' own text: encoding them again would shuffle their keys
        return ToolCall(name: name, argumentsJson: JSONText.member("arguments", of: body) ?? "{}")
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
    /// the shapes chat templates render natively. Tool arguments and results
    /// that are JSON go in as recorded.
    static func engineMessagesJson(_ messages: [Message]) -> String? {
        guard !messages.isEmpty else { return nil }
        let items = messages.map { message -> String in
            switch message.role {
            case .tool:
                let response = jsonValue(message.content) != nil ? message.content : JSONText.literal(message.content)
                return "{\"role\":\"tool\",\"content\":[" +
                       "\(toolResponseJson(name: message.toolName ?? "", response: response))]}"
            case .model:
                var item = "{\"role\":\"assistant\",\"content\":\(JSONText.literal(message.content))"
                if let calls = message.toolCalls, !calls.isEmpty {
                    let encoded = calls.map { call -> String in
                        let arguments = jsonValue(call.argumentsJson) != nil ? call.argumentsJson : "{}"
                        return "{\"type\":\"function\",\"function\":{\"name\":\(JSONText.literal(call.name))," +
                               "\"arguments\":\(arguments)}}"
                    }
                    item += ",\"tool_calls\":[\(encoded.joined(separator: ","))]"
                }
                return item + "}"
            default:
                return "{\"role\":\(JSONText.literal(message.role.stringValue))," +
                       "\"content\":\(JSONText.literal(message.content))}"
            }
        }
        return "[\(items.joined(separator: ","))]"
    }

    /// `{"type":"tool_response","name":…,"response":…}`, with the response's JSON as given.
    private static func toolResponseJson(name: String, response: String) -> String {
        "{\"type\":\"tool_response\",\"name\":\(JSONText.literal(name)),\"response\":\(response)}"
    }

    private static func jsonValue(_ json: String) -> Any? {
        guard let data = json.data(using: .utf8) else { return nil }
        return try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
    }
}

/// JSON written and read as text, so keys keep their order. The readers are
/// for text JSONSerialization has already accepted: they find where each value
/// ends and return its text, without validating it.
enum JSONText {

    /// `string` as a JSON string literal.
    static func literal(_ string: String) -> String {
        var out = "\""
        for scalar in string.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case _ where scalar.value < 0x20: out += String(format: "\\u%04x", scalar.value)
            default: out.unicodeScalars.append(scalar)
            }
        }
        return out + "\""
    }

    /// The text of the value named `key` in a JSON object (the last one, if the
    /// key repeats, as JSONSerialization reads it); nil when there's none or
    /// `json` isn't an object.
    static func member(_ key: String, of json: String) -> String? {
        members(of: json)?.last { $0.key == key }?.value
    }

    /// Each member of a JSON object as its key and its value's text, in order;
    /// nil when `json` isn't an object.
    static func members(of json: String) -> [(key: String, value: String)]? {
        var scanner = Cursor(json)
        guard scanner.take(.openBrace) else { return nil }
        var members: [(key: String, value: String)] = []
        if scanner.take(.closeBrace) { return members }
        repeat {
            guard let key = scanner.string().flatMap(decodedString), scanner.take(.colon),
                  let value = scanner.value() else { return nil }
            members.append((key, value))
        } while scanner.take(.comma)
        return scanner.take(.closeBrace) ? members : nil
    }

    /// The text of each element of a JSON array, in order; nil when `json`
    /// isn't an array.
    static func elements(of json: String) -> [String]? {
        var scanner = Cursor(json)
        guard scanner.take(.openBracket) else { return nil }
        var elements: [String] = []
        if scanner.take(.closeBracket) { return elements }
        repeat {
            guard let element = scanner.value() else { return nil }
            elements.append(element)
        } while scanner.take(.comma)
        return scanner.take(.closeBracket) ? elements : nil
    }

    /// A string literal's value.
    private static func decodedString(_ literal: String) -> String? {
        guard literal.contains("\\") else { return String(literal.dropFirst().dropLast()) }
        return (try? JSONSerialization.jsonObject(with: Data(literal.utf8), options: [.fragmentsAllowed])) as? String
    }

    /// Walks the UTF-8 of JSON text. Every byte it looks for is ASCII, which
    /// never occurs inside a multi-byte character.
    private struct Cursor {
        private static let whitespace: Set<UInt8> = [0x20, 0x09, 0x0A, 0x0D]
        private static let delimiters: Set<UInt8> = [0x20, 0x09, 0x0A, 0x0D, .comma, .colon, .closeBrace, .closeBracket]

        private let bytes: [UInt8]
        private var at = 0

        init(_ text: String) { bytes = Array(text.utf8) }

        /// Skips whitespace, then consumes `byte` if it comes next.
        mutating func take(_ byte: UInt8) -> Bool {
            skipSpace()
            guard at < bytes.count, bytes[at] == byte else { return false }
            at += 1
            return true
        }

        /// Skips whitespace, then consumes a string literal and returns its text.
        mutating func string() -> String? {
            skipSpace()
            guard at < bytes.count, bytes[at] == .quote else { return nil }
            let start = at
            guard skipString() else { return nil }
            return String(decoding: bytes[start..<at], as: UTF8.self)
        }

        /// Skips whitespace, then consumes a value and returns its text.
        mutating func value() -> String? {
            skipSpace()
            let start = at
            guard skipValue() else { return nil }
            return String(decoding: bytes[start..<at], as: UTF8.self)
        }

        private mutating func skipSpace() {
            while at < bytes.count, Self.whitespace.contains(bytes[at]) { at += 1 }
        }

        /// Moves past the value here: a string; an object or array, to its
        /// matching close; or a number or literal, to the next delimiter.
        private mutating func skipValue() -> Bool {
            guard at < bytes.count else { return false }
            switch bytes[at] {
            case .quote:
                return skipString()
            case .openBrace, .openBracket:
                var depth = 0
                while at < bytes.count {
                    switch bytes[at] {
                    case .quote:
                        guard skipString() else { return false }
                        continue
                    case .openBrace, .openBracket:
                        depth += 1
                    case .closeBrace, .closeBracket:
                        depth -= 1
                        if depth == 0 {
                            at += 1
                            return true
                        }
                    default:
                        break
                    }
                    at += 1
                }
                return false
            default:
                let start = at
                while at < bytes.count, !Self.delimiters.contains(bytes[at]) { at += 1 }
                return at > start
            }
        }

        /// Moves past the string literal that opens here, escapes and all.
        private mutating func skipString() -> Bool {
            at += 1
            while at < bytes.count {
                let byte = bytes[at]
                at += byte == .backslash ? 2 : 1
                if byte == .quote { return true }
            }
            return false
        }
    }
}

private extension UInt8 {
    static let quote = UInt8(ascii: "\""), backslash = UInt8(ascii: "\\")
    static let openBrace = UInt8(ascii: "{"), closeBrace = UInt8(ascii: "}")
    static let openBracket = UInt8(ascii: "["), closeBracket = UInt8(ascii: "]")
    static let comma = UInt8(ascii: ","), colon = UInt8(ascii: ":")
}
