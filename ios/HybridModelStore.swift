//
//  HybridModelStore.swift
//  react-native-litert-lm
//
//  Created by Antigravity on 2026-06-04.
//  Copyright © 2026 Margelo. All rights reserved.
//

import Foundation
import NitroModules

public class HybridModelStore: HybridModelStoreSpec_base, HybridModelStoreSpec_protocol {
    
    private let queue = DispatchQueue(label: "dev.litert.modelstore", qos: .utility)
    
    private var modelsDirectory: String {
        let cachesDir = NSSearchPathForDirectoriesInDomains(.cachesDirectory, .userDomainMask, true).first ?? NSTemporaryDirectory()
        return (cachesDir as NSString).appendingPathComponent("litert_models")
    }
    
    private func sanitizeFileName(_ fileName: String) throws {
        if fileName.contains("..") || fileName.contains("/") || fileName.contains("\\") {
            throw NSError(
                domain: "LiteRTLM.ModelStore",
                code: 400,
                userInfo: [NSLocalizedDescriptionKey: "Invalid filename: path traversal or directory separators are not allowed."]
            )
        }
    }
    
    public func isCached(fileName: String) throws -> Bool {
        try sanitizeFileName(fileName)
        let filePath = (modelsDirectory as NSString).appendingPathComponent(fileName)
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: filePath) {
            let attrs = try? fileManager.attributesOfItem(atPath: filePath)
            if let fileSize = attrs?[.size] as? UInt64, fileSize > 0 {
                return true
            }
        }
        return false
    }
    
    public func getFilePath(fileName: String) throws -> String {
        try sanitizeFileName(fileName)
        return (modelsDirectory as NSString).appendingPathComponent(fileName)
    }

    public func getFileSizeBytes(absolutePath: String) throws -> Double {
        let attrs = try? FileManager.default.attributesOfItem(atPath: absolutePath)
        if let size = attrs?[.size] as? UInt64 {
            return Double(size)
        }
        return -1.0
    }
    
    public func listCachedFiles() throws -> [ModelFile] {
        let fileManager = FileManager.default
        let dir = modelsDirectory
        
        guard fileManager.fileExists(atPath: dir) else {
            return []
        }
        
        let contents = try fileManager.contentsOfDirectory(atPath: dir)
        var files: [ModelFile] = []
        
        for name in contents {
            if name.hasSuffix(".tmp") { continue }
            let path = (dir as NSString).appendingPathComponent(name)
            var isDir: ObjCBool = false
            if fileManager.fileExists(atPath: path, isDirectory: &isDir), !isDir.boolValue {
                let attrs = try? fileManager.attributesOfItem(atPath: path)
                let size = (attrs?[.size] as? UInt64) ?? 0
                let modDate = (attrs?[.modificationDate] as? Date) ?? Date()
                
                files.append(ModelFile(
                    fileName: name,
                    absolutePath: path,
                    sizeBytes: Double(size),
                    lastModifiedMs: modDate.timeIntervalSince1970 * 1000.0
                ))
            }
        }
        return files
    }
    
    public func deleteFile(fileName: String) throws {
        try sanitizeFileName(fileName)
        let filePath = (modelsDirectory as NSString).appendingPathComponent(fileName)
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: filePath) {
            try fileManager.removeItem(atPath: filePath)
        }
    }
    
    public func downloadFile(
        url: String,
        fileName: String,
        headersJson: String,
        onProgress: @escaping (_ progress: Double) -> Void
    ) throws -> Promise<String> {
        let promise = Promise<String>()
        
        queue.async {
            do {
                try self.sanitizeFileName(fileName)
                let destPath = (self.modelsDirectory as NSString).appendingPathComponent(fileName)
                let fileManager = FileManager.default
                
                if !fileManager.fileExists(atPath: self.modelsDirectory) {
                    try fileManager.createDirectory(atPath: self.modelsDirectory, withIntermediateDirectories: true, attributes: nil)
                }
                
                // Fast cache check
                if fileManager.fileExists(atPath: destPath) {
                    let attrs = try fileManager.attributesOfItem(atPath: destPath)
                    if let fileSize = attrs[.size] as? UInt64, fileSize > 0 {
                        onProgress(1.0)
                        promise.resolve(withResult: destPath)
                        return
                    }
                }
                
                guard let downloadUrl = URL(string: url), downloadUrl.scheme?.lowercased() == "https" else {
                    promise.reject(withError: NSError(domain: "LiteRTLM.ModelStore", code: 400, userInfo: [NSLocalizedDescriptionKey: "Invalid download URL: HTTPS is required for security."]))
                    return
                }
                
                onProgress(0.0)
                
                // Parse headers JSON
                var customHeaders: [String: String] = [:]
                if !headersJson.isEmpty {
                    if let data = headersJson.data(using: .utf8),
                       let parsed = try? JSONSerialization.jsonObject(with: data, options: []) as? [String: String] {
                        customHeaders = parsed
                    }
                }
                
                let sessionConfig = URLSessionConfiguration.default
                sessionConfig.timeoutIntervalForRequest = 30
                sessionConfig.timeoutIntervalForResource = 3600
                
                let session = URLSession(configuration: sessionConfig)
                var progressHandler: NSKeyValueObservation?
                
                var request = URLRequest(url: downloadUrl)
                for (key, value) in customHeaders {
                    request.addValue(value, forHTTPHeaderField: key)
                }
                
                let tempDestFile = (self.modelsDirectory as NSString).appendingPathComponent("\(fileName).tmp")
                
                let task = session.downloadTask(with: request) { location, response, error in
                    progressHandler?.invalidate()
                    
                    if let error = error {
                        promise.reject(withError: error)
                        return
                    }
                    
                    if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode >= 400 {
                        promise.reject(withError: NSError(domain: "LiteRTLM.ModelStore", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP \(httpResponse.statusCode)"]))
                        return
                    }
                    
                    guard let location = location else {
                        promise.reject(withError: NSError(domain: "LiteRTLM.ModelStore", code: 500, userInfo: [NSLocalizedDescriptionKey: "No download location found."]))
                        return
                    }
                    
                    do {
                        // Rename/move atomically via temp file helper
                        if fileManager.fileExists(atPath: tempDestFile) {
                            try fileManager.removeItem(atPath: tempDestFile)
                        }
                        try fileManager.moveItem(at: location, to: URL(fileURLWithPath: tempDestFile))
                        
                        if fileManager.fileExists(atPath: destPath) {
                            try fileManager.removeItem(atPath: destPath)
                        }
                        try fileManager.moveItem(at: URL(fileURLWithPath: tempDestFile), to: URL(fileURLWithPath: destPath))
                        
                        onProgress(1.0)
                        promise.resolve(withResult: destPath)
                    } catch {
                        promise.reject(withError: error)
                    }
                }
                
                var lastUpdate = Date()
                progressHandler = task.observe(\.countOfBytesReceived, options: [.new]) { task, _ in
                    let expected = task.countOfBytesExpectedToReceive
                    if expected > 0 {
                        let now = Date()
                        if now.timeIntervalSince(lastUpdate) > 0.1 {
                            let progress = Double(task.countOfBytesReceived) / Double(expected)
                            onProgress(progress)
                            lastUpdate = now
                        }
                    }
                }
                
                task.resume()
                session.finishTasksAndInvalidate()
            } catch {
                promise.reject(withError: error)
            }
        }
        
        return promise
    }
}
