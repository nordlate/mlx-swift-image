import FluxKlein

public extension KleinGenerationPixels {
    static func budget(native: [UInt8], reference: [UInt8]) -> (max: Int, mean: Double, passed: Bool) {
        guard native.count == reference.count, !native.isEmpty else { return (256, .infinity, false) }
        let diffs = zip(native,reference).map { abs(Int($0)-Int($1)) }
        let maximum = diffs.max()!, mean = Double(diffs.reduce(0,+))/Double(diffs.count)
        return (maximum,mean,Double(maximum) <= KleinGenerationBudgets.pixelMax && mean <= KleinGenerationBudgets.pixelMean)
    }
}
