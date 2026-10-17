import FluxKlein
import Foundation

public struct KleinStartingImageCase: Codable, Equatable, Sendable {
    public let id: String, prompt: String
    public let width: Int, height: Int, seed: UInt64, steps: Int
    public let strength: Double
    public var tiny: Bool { steps == 0 }
    public var start: Int { tiny ? 0 : max(1,Int(Double(steps)*strength)) }
}
public enum KleinStartingImageCases {
    public static let algorithm = "RGB-v1:R=(3*x+5*y+17*seed)%256;G=(7*x+11*y+13*seed)%256;B=(19*x+23*y+3*seed)%256"
    public static let all: [KleinStartingImageCase] = [
        .init(id: "i1-square768-mid",prompt: "A red bicycle leaning against a brick wall.",width: 768,height: 768,seed: 42,steps: 4,strength: 0.625),
        .init(id: "i2-portrait672x896-many",prompt: "A small cabin beside a Finnish lake at sunrise.",width: 672,height: 896,seed: 7,steps: 8,strength: 0.1875),
        .init(id: "i3-landscape896x672-one",prompt: "A warm lantern on a wooden table.",width: 896,height: 672,seed: 31337,steps: 8,strength: 0.9375),
        .init(id: "i4-square1024-min",prompt: "A blue ceramic bowl.",width: 1024,height: 1024,seed: 0,steps: 2,strength: 0.75),
        .init(id: "e1-rect96x64",prompt: "",width: 96,height: 64,seed: 19,steps: 0,strength: 0),
        .init(id: "e2-square64",prompt: "",width: 64,height: 64,seed: 0,steps: 0,strength: 0),
    ]
    public static var allIDs: [String] { all.map(\.id) }
    public static var digest: String {
        let lines = all.map { [$0.id,String($0.width),String($0.height),String($0.seed),String($0.steps),String($0.strength),KleinHashing.sha256(of: Data($0.prompt.utf8))].joined(separator: "|") }
        return KleinHashing.sha256(of: Data(([algorithm]+lines).joined(separator: "\n").utf8))
    }
    public static func check(_ url: URL) throws {
        let data = try Data(contentsOf: url), decoded = try JSONDecoder().decode([KleinStartingImageCase].self,from: data)
        let raw = try JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        guard decoded == all, raw?.allSatisfy({ Set($0.keys) == Set(["id","prompt","width","height","seed","steps","strength"]) }) == true else {
            throw KleinTransformerError.invalidInput("starting-image corpus differs from compiled four primary/two encoder cases")
        }
    }
    public static func rgb(_ c: KleinStartingImageCase) -> [UInt8] {
        var bytes: [UInt8] = []; bytes.reserveCapacity(c.width*c.height*3)
        let seed = Int(c.seed % 256)
        for y in 0..<c.height { for x in 0..<c.width {
            bytes.append(UInt8((3*x+5*y+17*seed)%256)); bytes.append(UInt8((7*x+11*y+13*seed)%256)); bytes.append(UInt8((19*x+23*y+3*seed)%256))
        } }
        return bytes
    }
}
public enum KleinStartingImageBudgets {
    public static let component = 1e-3, encoded = 2e-3, clean = 2e-3, postStep = 5e-3
    public static let expected: [String: Any] = [
        "relative_l2":["encoder_component":component,"encoded":encoded,"same_input_patch_bn_pack":component,"clean_initial":clean,"post_step":postStep,"prompt_embeds":1e-3,"isolated_decoder":2e-3],
        "prompt_embeds_normalized_max_abs":1e-2,"zero_reference_absolute":KleinGenerationBudgets.zero,
        "pixels_max_abs":2.0,"pixels_mean_abs":0.1,"weights":"exact","source_rgb":"exact","prepared":"exact",
        "seeded_noise_grid_schedule_start_callbacks":"exact","conversion":"exact","png_readback":"exact","non_finite":"fail",
        "noise":"diagnostic","full_pipeline_decoded":"diagnostic"]
    public static func check(_ url: URL) throws {
        guard let actual = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? NSDictionary, actual.isEqual(to: expected) else {
            throw KleinTransformerError.invalidInput("starting-image budgets differ from predeclared compiled gates")
        }
    }
}
public struct KleinStartingImageFixture: Codable, Equatable, Sendable {
    public let id: String
    public let width: Int, height: Int
    public let seed: UInt64
    public let png_sha256: String, rgb_sha256: String
}
public struct KleinStartingImageFixtures: Codable, Sendable {
    public let schema: Int
    public let algorithm: String
    public let cases: [KleinStartingImageFixture]
    public static func check(_ url: URL) throws -> [String: String] {
        let manifest = try KleinJSON.read(Self.self,from: url)
        guard manifest.schema == 1, manifest.algorithm == KleinStartingImageCases.algorithm,
              manifest.cases.map(\.id) == KleinStartingImageCases.allIDs else { throw KleinTransformerError.invalidInput("fixture manifest corpus/algorithm differs") }
        var hashes: [String: String] = [:]
        for (entry,c) in zip(manifest.cases,KleinStartingImageCases.all) {
            let image = try KleinPreparedStartingImage.load(url.deletingLastPathComponent().appendingPathComponent(c.id+".png"),width: c.width,height: c.height)
            guard entry.width == c.width, entry.height == c.height, entry.seed == c.seed,
                  entry.png_sha256 == image.fileSHA256, entry.rgb_sha256 == image.rgbSHA256, image.rgb == KleinStartingImageCases.rgb(c) else {
                throw KleinTransformerError.artifactChanged("fixture RGB/file pin differs: \(c.id)")
            }
            hashes[c.id] = image.fileSHA256
        }
        return hashes
    }
}
public enum KleinStartingImageFreeze {
    public static func values(casesJSON: URL,budgetsJSON: URL,fixtures: URL) throws -> [String: String] {
        try KleinStartingImageCases.check(casesJSON); try KleinStartingImageBudgets.check(budgetsJSON)
        _ = try KleinStartingImageFixtures.check(fixtures)
        return ["cases_sha256":try KleinHashing.sha256(of: casesJSON),"budgets_sha256":try KleinHashing.sha256(of: budgetsJSON),
                "fixtures_sha256":try KleinHashing.sha256(of: fixtures),"cases_digest":KleinStartingImageCases.digest]
    }
    public static func record(casesJSON: URL,budgetsJSON: URL,fixtures: URL) throws -> String {
        String(decoding: try JSONSerialization.data(withJSONObject: values(casesJSON: casesJSON,budgetsJSON: budgetsJSON,fixtures: fixtures),options: [.prettyPrinted,.sortedKeys]),as: UTF8.self)
    }
    static func require(_ url: URL,cases: URL,budgets: URL,fixtures: URL) throws -> [String: String] {
        let frozen = try values(casesJSON: cases,budgetsJSON: budgets,fixtures: fixtures)
        guard try KleinJSON.read([String: String].self,from: url) == frozen else { throw KleinTransformerError.invalidInput("freeze differs; freeze before numerics") }
        return frozen
    }
}
public enum KleinStartingImagePaths {
    public static func requireNew(_ url: URL,model: URL? = nil) throws {
        try KleinGenerationPaths.requireNew(url,modelDirectory: model)
        if let model {
            // Resource identity, rather than string case folding, handles case-insensitive APFS aliases.
            let fm = FileManager.default, target = model.resolvingSymlinksInPath()
            let targetAttributes = try fm.attributesOfItem(atPath: target.path)
            var prefix = url.deletingLastPathComponent()
            while prefix.path != "/" {
                if fm.fileExists(atPath: prefix.path) {
                    let attributes = try fm.attributesOfItem(atPath: prefix.path)
                    if attributes[.systemNumber] as? NSNumber == targetAttributes[.systemNumber] as? NSNumber,
                       attributes[.systemFileNumber] as? NSNumber == targetAttributes[.systemFileNumber] as? NSNumber {
                        throw KleinTransformerError.invalidInput("output ancestor is the model directory")
                    }
                }
                prefix.deleteLastPathComponent()
            }
        }
    }
    /// Creates `url` as a NEW directory: the same freshness, symlink and model-identity checks as `requireNew`, missing parents
    /// created, then a POSIX `mkdir` of the leaf, which fails atomically with EEXIST. `FileManager.createDirectory(...,
    /// withIntermediateDirectories: true)` would silently reuse a leaf that appeared after the check.
    public static func createNewDirectory(_ url: URL,model: URL? = nil) throws {
        try requireNew(url,model: model)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),withIntermediateDirectories: true)
        guard mkdir(url.path,0o755) == 0 else {
            let code = errno
            throw KleinTransformerError.invalidInput(code == EEXIST ? "output already exists: \(url.path)" : "cannot create output directory \(url.path): \(String(cString: strerror(code)))")
        }
    }
}
