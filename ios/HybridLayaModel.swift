//
//  HybridLayaModel.swift
//  react-native-litert-lm
//

import Foundation
import NitroModules

public class HybridLayaModel: HybridLayaModelSpec_base, HybridLayaModelSpec_protocol {

    private var isLoaded: Bool = false
    private let queue = DispatchQueue(label: "dev.litert.laya", qos: .userInitiated)

    public override init() {
        super.init()
    }

    public func loadModel(modelDir: String, config: LayaModelConfig?) throws -> Promise<Void> {
        return Promise.async {
            self.queue.sync {
                let cleanPath = modelDir.hasPrefix("file://") ? String(modelDir.dropFirst(7)) : modelDir
                var isDir: ObjCBool = false
                guard FileManager.default.fileExists(atPath: cleanPath, isDirectory: &isDir), isDir.boolValue else {
                    fatalError("Laya model directory not found at: \(cleanPath)")
                }
                self.isLoaded = true
            }
        }
    }

    public func decide(request: LayaDecisionRequest) throws -> Promise<LayaDecisionResponse> {
        return Promise.async {
            return self.queue.sync {
                guard self.isLoaded else {
                    fatalError("Laya model is not loaded. Call loadModel() first.")
                }

                let startTime = CFAbsoluteTimeGetCurrent()

                var choice: String? = nil
                var score: Double? = nil
                var noul: Double? = nil
                var decision: Bool? = nil
                var probabilities: [String: Double] = [:]

                let qtype = request.type.lowercased()
                if qtype == "choice" {
                    if let opts = request.options, !opts.isEmpty {
                        choice = opts[0]
                        let prob = 1.0 / Double(opts.count)
                        for opt in opts {
                            probabilities[opt] = prob
                        }
                    } else {
                        choice = "default"
                        probabilities["default"] = 1.0
                    }
                } else if qtype == "score" {
                    score = 0.0
                    probabilities["0"] = 1.0
                } else if qtype == "noul" {
                    noul = 0.5
                    decision = true
                    probabilities["false"] = 0.5
                    probabilities["true"] = 0.5
                }

                let latencyMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0

                return LayaDecisionResponse(
                    type: request.type,
                    choice: choice,
                    score: score,
                    noul: noul,
                    decision: decision,
                    probabilities: probabilities,
                    confidence: 1.0,
                    actProbability: 1.0,
                    latencyMs: latencyMs
                )
            }
        }
    }

    public func isReady() throws -> Bool {
        return queue.sync { isLoaded }
    }

    public func unload() throws -> Promise<Void> {
        return Promise.async {
            self.queue.sync {
                self.isLoaded = false
            }
        }
    }

    public func close() throws -> Void {
        queue.sync {
            self.isLoaded = false
        }
    }
}
