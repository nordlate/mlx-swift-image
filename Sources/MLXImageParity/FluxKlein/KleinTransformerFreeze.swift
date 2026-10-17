import FluxKlein
import Foundation

/// The freeze record: what must be written to `klein/transformer/results/frozen-inputs.json` before the first real reference or
/// native run. The comparison fails when any of these differs from what the runs used.
public enum KleinTransformerFreeze {
    /// Validates both files against the compiled tables and returns the JSON of the three pins (sorted keys).
    public static func record(casesJSON: URL, budgetsJSON: URL) throws -> String {
        let (cases, casesSHA) = try KleinTransformerCases.load(url: casesJSON)
        let failures = KleinTransformerCases.structuralFailures(cases) + (try KleinTransformerBudgets.check(budgetsJSON: budgetsJSON))
        guard failures.isEmpty else { throw KleinTransformerError.invalidInput("cannot freeze: \(failures.joined(separator: "; "))") }
        let record: [String: String] = [
            "budgets_sha256": try KleinHashing.sha256(of: budgetsJSON), "cases_digest": KleinTransformerCases.digest(of: cases), "cases_sha256": casesSHA,
        ]
        let data = try JSONSerialization.data(withJSONObject: record, options: [.prettyPrinted, .sortedKeys])
        return String(decoding: data, as: UTF8.self)
    }
}
