//
// HybridLiteRTLM+Products.swift
//
// The product-template and tuning APIs are implemented on Android only for
// now. These stubs keep the iOS target conforming to the regenerated spec;
// each rejects (or throws) with a clear message instead of failing the build.
//

import Foundation
import NitroModules

extension HybridLiteRTLM {
  private static func androidOnly(_ name: String) -> Error {
    RuntimeError.error(withMessage: "LiteRTLM.\(name)() is not implemented on iOS yet (Android only).")
  }

  func registerProduct(productId: String, systemPrompt: String, options: ProductOptions?) throws -> Promise<Void> {
    Promise.rejected(withError: Self.androidOnly("registerProduct"))
  }

  func prewarmProducts(productIds: [String]) throws -> Promise<Void> {
    Promise.rejected(withError: Self.androidOnly("prewarmProducts"))
  }

  func executeProduct(productId: String, text: String,
                      onToken: ((_ token: String, _ done: Bool) -> Void)?,
                      options: ProductExecuteOptions?) throws -> Promise<String> {
    Promise.rejected(withError: Self.androidOnly("executeProduct"))
  }

  func cancelProduct() throws {}

  func evictProducts(productIds: [String]?) throws -> Promise<Void> {
    Promise.resolved()
  }

  func getProductStats() throws -> Promise<ProductStats> {
    Promise.resolved(withResult: ProductStats(requests: 0, drafterFailures: 0, recoveredByRetry: 0,
                                              productsWithMtpOff: [], residentProducts: []))
  }

  func contextTokensFor(longestSystemPromptChars: Double, maxUserChars: Double, maxOutputChars: Double) throws -> Double {
    // Same formula as Android: 3 chars/token margin + 64, rounded up to 1K, 1K-32K.
    let tokens = (longestSystemPromptChars + maxUserChars + maxOutputChars) / 3 + 64
    return min(max((tokens / 1024).rounded(.up) * 1024, 1024), 32768)
  }

  func tune(modelPath: String, maxContextTokens: Double,
            onProgress: ((_ measurement: TuneMeasurement) -> Void)?) throws -> Promise<TunedConfig> {
    Promise.rejected(withError: Self.androidOnly("tune"))
  }

  func getTunedConfig(modelPath: String) throws -> TunedConfig? { nil }
}
