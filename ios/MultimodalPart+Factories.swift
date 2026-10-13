//
//  MultimodalPart+Factories.swift
//  react-native-litert-lm
//
//  Mirrors src/inferenceRouting.ts — keep shapes in sync for native direct hybrid access.
//

import NitroModules

extension MultimodalPart {
    static func textPart(_ text: String) -> MultimodalPart {
        MultimodalPart(type: .text, text: text, path: nil, imageBuffer: nil, audioBuffer: nil, name: nil, responseJson: nil)
    }

    static func imagePart(_ path: String) -> MultimodalPart {
        MultimodalPart(type: .image, text: nil, path: path, imageBuffer: nil, audioBuffer: nil, name: nil, responseJson: nil)
    }

    static func audioPart(_ path: String) -> MultimodalPart {
        MultimodalPart(type: .audio, text: nil, path: path, imageBuffer: nil, audioBuffer: nil, name: nil, responseJson: nil)
    }
}
