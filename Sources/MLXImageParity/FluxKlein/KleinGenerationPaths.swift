import FluxKlein
import Foundation

public extension KleinGenerationPaths {
    static func create(_ url: URL, modelDirectory: URL) throws {
        try requireNew(url,modelDirectory: modelDirectory)
        try FileManager.default.createDirectory(at: url,withIntermediateDirectories: true)
    }
}
