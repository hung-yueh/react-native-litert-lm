package com.margelo.nitro.dev.litert.litertlm

import android.util.Log
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.google.gson.JsonPrimitive
import org.json.JSONObject
import com.google.ai.edge.litertlm.ToolCall as SdkToolCall

/*
 * The tool-calling round trip: a message of tool results in, structured tool
 * calls out of the transcript, and SDK messages for replaying a transcript
 * into a new conversation. Mirrors ios/HybridLiteRTLM+Tools.swift.
 */

private const val TAG = "LiteRTLM.Tools"
private const val CALL_START = "<tool_call>"
private const val CALL_END = "</tool_call>"

/** A 'tool' message answering the model's calls; each result's JSON passes through unchanged. */
internal fun toolResultMessage(results: List<Pair<String, String>>): LiteRTMessage =
    LiteRTMessage.tool(
        Contents.of(
            results.map { (name, json) ->
                val response = try {
                    JsonParser.parseString(json)
                } catch (e: Exception) {
                    throw LiteRTLMNativeException(
                        "execute", "INVALID_ARGUMENT", "INVALID_INPUT",
                        "The result for tool '$name' is not valid JSON", e,
                    )
                }
                Content.ToolResponse(name, response)
            },
        ),
    )

/** The arguments of SDK-parsed tool calls as Nitro [ToolCall]s (null when none). */
internal fun List<SdkToolCall>?.toNitroToolCalls(): Array<ToolCall>? =
    this?.takeIf { it.isNotEmpty() }
        ?.map { ToolCall(it.name, JSONObject(it.arguments).toString()) }
        ?.toTypedArray()

/**
 * Splits `<tool_call>{"name":…,"arguments":{…}}</tool_call>` markers out of a
 * response: the remaining text, and the calls. A marker whose JSON doesn't
 * parse stays in the text.
 */
internal fun splitToolCallMarkers(response: String): Pair<String, List<ToolCall>> {
    if (!response.contains(CALL_START)) return response to emptyList()
    val calls = mutableListOf<ToolCall>()
    val text = StringBuilder()
    var rest = response
    while (true) {
        val start = rest.indexOf(CALL_START)
        val end = if (start < 0) -1 else rest.indexOf(CALL_END, start + CALL_START.length)
        if (end < 0) break
        text.append(rest, 0, start)
        val call = parseToolCall(rest.substring(start + CALL_START.length, end))
        if (call != null) calls.add(call) else text.append(rest, start, end + CALL_END.length)
        rest = rest.substring(end + CALL_END.length)
    }
    text.append(rest)
    return text.toString().trim() to calls
}

private fun parseToolCall(body: String): ToolCall? {
    val obj = (try { JsonParser.parseString(body) } catch (e: Exception) { null })
        ?.takeIf { it.isJsonObject }?.asJsonObject ?: return null
    val name = obj.string("name") ?: return null
    val arguments = obj.get("arguments")?.takeIf { it.isJsonObject } ?: JsonObject()
    return ToolCall(name, arguments.toString())
}

/**
 * A transcript from JS (the JSON of `getHistory()`) as Messages. An unknown
 * role reads as a user turn; "assistant" reads as the model.
 */
internal fun parseTranscript(json: String?): List<Message> {
    if (json.isNullOrEmpty()) return emptyList()
    val items = try {
        JsonParser.parseString(json).asJsonArray
    } catch (e: Exception) {
        Log.e(TAG, "resetConversation: unparseable historyJson — starting empty", e)
        return emptyList()
    }
    return items.mapNotNull { element ->
        val item = element.takeIf { it.isJsonObject }?.asJsonObject ?: return@mapNotNull null
        val content = item.string("content") ?: return@mapNotNull null
        val role = when (item.string("role")) {
            "model", "assistant" -> Role.MODEL
            "system" -> Role.SYSTEM
            "tool" -> Role.TOOL
            else -> Role.USER
        }
        val calls = item.get("toolCalls")?.takeIf { it.isJsonArray }?.asJsonArray?.mapNotNull { call ->
            val obj = call.takeIf { it.isJsonObject }?.asJsonObject ?: return@mapNotNull null
            ToolCall(obj.string("name") ?: return@mapNotNull null, obj.string("argumentsJson") ?: "{}")
        }.orEmpty()
        Message(role, content, calls.takeIf { it.isNotEmpty() }?.toTypedArray(), item.string("toolName"))
    }
}

/** The SDK message that replays one transcript turn, tool exchanges included. */
internal fun Message.toSdkMessage(): LiteRTMessage = when (role) {
    Role.MODEL -> LiteRTMessage.model(
        contents = if (content.isEmpty()) Contents.of(emptyList()) else Contents.of(content),
        toolCalls = toolCalls.orEmpty().map { call ->
            val arguments = lenientJson(call.argumentsJson).takeIf { it.isJsonObject }?.asJsonObject ?: JsonObject()
            SdkToolCall(call.name, arguments.entrySet().associate { (key, value) -> key to value })
        },
    )
    Role.TOOL -> LiteRTMessage.tool(Contents.of(Content.ToolResponse(toolName ?: "", lenientJson(content))))
    Role.SYSTEM -> LiteRTMessage.system(content)
    Role.USER -> LiteRTMessage.user(content)
}

/** JSON when [text] parses, else the text itself as a JSON string. */
private fun lenientJson(text: String): JsonElement =
    try { JsonParser.parseString(text) } catch (e: Exception) { JsonPrimitive(text) }

private fun JsonObject.string(key: String): String? =
    get(key)?.takeIf { it.isJsonPrimitive }?.asString
