import Foundation
import MLXImageCore

public struct KleinGenerationCase: Equatable, Sendable {
    public let id: String, category: String, prompt: String
    public let width: Int, height: Int
    public let seed: UInt64
    public let steps: Int
}
public struct KleinGenerationVAECase: Equatable, Sendable {
    public let id: String
    public let width: Int, height: Int
    public let seed: UInt64
}
public enum KleinGenerationCases {
    public static let frozen = KleinTransformerCases.frozen.map {
        KleinGenerationCase(id: "g" + String($0.id.dropFirst()), category: $0.category, prompt: $0.prompt,
                            width: $0.width, height: $0.height, seed: $0.seed, steps: $0.steps)
    }
    public static let vae = [KleinGenerationVAECase(id: "v1-rect96x64-seeded", width: 96, height: 64, seed: 19),
                             KleinGenerationVAECase(id: "v2-square64-seeded", width: 64, height: 64, seed: 0)]
    public static let allIDs = frozen.map(\.id) + vae.map(\.id)
    public static var digest: String {
        let primary = frozen.map { [$0.id,$0.category,String($0.width),String($0.height),String($0.seed),String($0.steps),KleinHashing.sha256(of: Data($0.prompt.utf8))].joined(separator: "|") }
        let synthetic = vae.map { [$0.id,String($0.width),String($0.height),String($0.seed)].joined(separator: "|") }
        return KleinHashing.sha256(of: Data((primary + synthetic).joined(separator: "\n").utf8))
    }
    public static func check(_ url: URL) throws {
        guard let root = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any],
              let cases = root["cases"] as? [[String: Any]], let small = root["vae_cases"] as? [[String: Any]],
              cases.count == frozen.count, small.count == vae.count else { throw KleinTransformerError.invalidInput("generation corpus incomplete") }
        let prompts = try KleinCases.parse(cases)
        for (i,c) in frozen.enumerated() {
            let e = cases[i]
            let keys: Set<String> = c.category == "long"
                ? ["id","category","prefix","unit","repeat","separator","width","height","seed","steps"]
                : ["id","category","prompt","width","height","seed","steps"]
            guard Set(e.keys) == keys else { throw KleinTransformerError.invalidInput("unknown or missing generation case fields") }
            guard prompts[i].id == c.id, prompts[i].category == c.category, Data(prompts[i].prompt.utf8) == Data(c.prompt.utf8),
                  e["width"] as? Int == c.width, e["height"] as? Int == c.height, e["seed"] as? UInt64 == c.seed,
                  e["steps"] as? Int == c.steps, e["timestep_index"] == nil else { throw KleinTransformerError.invalidInput("generation corpus differs: \(c.id)") }
            try KleinGenerationRequest(prompt: c.prompt, width: c.width, height: c.height, seed: c.seed, steps: c.steps).validate()
        }
        for (i,c) in vae.enumerated() {
            let e = small[i]
            guard Set(e.keys) == Set(["id","width","height","seed"]), e["id"] as? String == c.id,
                  e["width"] as? Int == c.width, e["height"] as? Int == c.height, e["seed"] as? UInt64 == c.seed else {
                throw KleinTransformerError.invalidInput("synthetic VAE corpus differs: \(c.id)")
            }
        }
    }
}
public enum KleinGenerationBudgets {
    public static let postStep = 5e-3, component = 1e-3, completeVAE = 2e-3
    public static let prompt = 1e-3, promptMax = 1e-2, pixelMax = 2.0, pixelMean = 0.1
    public static let zero = ParityTolerance.zeroFallbackAbsolute
    static var expected: [String: Any] { [
        "relative_l2": ["post_step":postStep,"vae_component":component,"vae_complete":completeVAE,"prompt_embeds":prompt],
        "prompt_embeds_normalized_max_abs":promptMax,"zero_reference_absolute":zero,"pixels_max_abs":pixelMax,"pixels_mean_abs":pixelMean,
        "non_finite":"fail","noise":"diagnostic","full_pipeline_decoded":"diagnostic","weights":"exact","conversion":"exact","png_readback":"exact"] }
    public static func check(_ url: URL) throws {
        let actual = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
        guard let a = actual as? NSDictionary, a.isEqual(to: expected) else { throw KleinTransformerError.invalidInput("generation budgets differ from compiled limits") }
    }
}
public enum KleinGenerationFreeze {
    public static func record(casesJSON: URL, budgetsJSON: URL) throws -> String {
        try KleinGenerationCases.check(casesJSON); try KleinGenerationBudgets.check(budgetsJSON)
        let r = ["cases_sha256":try KleinHashing.sha256(of: casesJSON),"budgets_sha256":try KleinHashing.sha256(of: budgetsJSON),"cases_digest":KleinGenerationCases.digest]
        return String(decoding: try JSONSerialization.data(withJSONObject: r, options: [.prettyPrinted,.sortedKeys]),as: UTF8.self)
    }
}

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
    static func create(_ url: URL, modelDirectory: URL) throws {
        try requireNew(url,modelDirectory: modelDirectory)
        try FileManager.default.createDirectory(at: url,withIntermediateDirectories: true)
    }
}
