import Foundation

/// Strict fresh paths: checks every lexical prefix, including dangling links and nonexistent tails. Pass a physical path.
public enum KleinGenerationPaths {
    public static func requireNew(_ url: URL, modelDirectory: URL? = nil) throws {
        let fm = FileManager.default
        guard url.path.hasPrefix("/"), !url.pathComponents.contains("..") else { throw KleinTransformerError.invalidInput("absolute physical path required") }
        var prefix = ""
        for c in url.path.split(separator: "/") {
            prefix += "/" + String(c)
            if (try? fm.destinationOfSymbolicLink(atPath: prefix)) != nil { throw KleinTransformerError.invalidInput("symbolic link in output path: \(prefix)") }
        }
        if let modelDirectory {
            let model = modelDirectory.resolvingSymlinksInPath().standardizedFileURL.path
            let path = url.standardizedFileURL.path
            guard path != model, !path.hasPrefix(model + "/") else { throw KleinTransformerError.invalidInput("output is inside model") }
        }
        guard !fm.fileExists(atPath: url.path) else { throw KleinTransformerError.invalidInput("output already exists: \(url.path)") }
    }
}
