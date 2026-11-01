//
//  HybridMetricModel.swift
//  react-native-litert-lm
//

import Foundation
import NitroModules

public class HybridMetricModel: HybridMetricModelSpec_base, HybridMetricModelSpec_protocol {
    
    private var isLoaded: Bool = false
    private let queue = DispatchQueue(label: "dev.litert.metric", qos: .userInitiated)
    
    public static let metricIndexMap: [String] = [
        "Warmth",                  // 0
        "Sentiment",               // 1
        "Directness",              // 2
        "Passive-Aggressive",      // 3
        "Chemistry",               // 4
        "Professional Impression", // 5
        "Urgency",                 // 6
        "Formality",               // 7
        "Tension",                 // 8
        "Empathy",                 // 9
        "Clarity",                 // 10
        "Assertiveness",           // 11
        "Playfulness"              // 12
    ]
    
    public override init() {
        super.init()
    }
    
    public func loadModel(modelPath: String, config: MetricModelConfig?) throws -> Promise<Void> {
        return Promise.async {
            self.queue.sync {
                let cleanPath = modelPath.hasPrefix("file://") ? String(modelPath.dropFirst(7)) : modelPath
                guard FileManager.default.fileExists(atPath: cleanPath) else {
                    fatalError("Metric model file not found at: \(cleanPath)")
                }
                self.isLoaded = true
            }
        }
    }
    
    public func evaluate(input: MetricEvaluationInput) throws -> Promise<MetricEvaluationResult> {
        return Promise.async {
            return self.queue.sync {
                guard self.isLoaded else {
                    fatalError("Metric model is not loaded. Call loadModel() first.")
                }
                
                let startTime = CFAbsoluteTimeGetCurrent()
                
                // Baseline neutral values (raw int8: 0, rescaled to 0.5)
                var scores: [String: Double] = [:]
                var values: [Double] = []
                var rawValues: [Double] = []
                
                for i in 0..<13 {
                    let axisName = HybridMetricModel.metricIndexMap[i]
                    let raw = 0.0
                    let normalized = 0.5
                    scores[axisName] = normalized
                    values.append(normalized)
                    rawValues.append(raw)
                }
                
                let latencyMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
                
                return MetricEvaluationResult(
                    scores: scores,
                    values: values,
                    rawValues: rawValues,
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
