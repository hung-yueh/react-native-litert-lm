package com.margelo.nitro.dev.litert.litertlm

/**
 * Mirrors [src/inferenceRouting.ts] — keep shapes in sync for native direct hybrid access.
 */
object MultimodalPartFactories {
    fun textPart(text: String): MultimodalPart =
        MultimodalPart(type = PartType.TEXT, text = text, path = null, imageBuffer = null, audioBuffer = null, name = null, responseJson = null)

    fun imagePart(path: String): MultimodalPart =
        MultimodalPart(type = PartType.IMAGE, text = null, path = path, imageBuffer = null, audioBuffer = null, name = null, responseJson = null)

    fun audioPart(path: String): MultimodalPart =
        MultimodalPart(type = PartType.AUDIO, text = null, path = path, imageBuffer = null, audioBuffer = null, name = null, responseJson = null)
}
