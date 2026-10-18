import Foundation

/// `manifest.json` of a bundle written by `tools/bundle/export.py`.
///
/// Every constant the PyTorch pipeline applies around the network -- variable order,
/// normalisation, imputation, bounding -- resolved to plain indices and numbers, so this
/// package never has to know anemoi's configuration format.
public struct Manifest: Codable, Sendable {
    public static let format = "aifs-mlx-bundle"
    public static let supportedVersion = 1

    public struct Processor: Codable, Sendable {
        public let layers: Int
        public let heads: Int
        public let window: Int
    }

    public struct Imputation: Codable, Sendable {
        /// Input column; NaNs there are replaced by `value` in normalised space.
        public let input: Int
        /// Matching output column, which gets the input's NaNs (at the first input time)
        /// put back after the forward pass. Nil for forcings, which are not predicted.
        public let output: Int?
        public let value: Float
    }

    public struct ConditionalNaN: Codable, Sendable {
        /// Output column whose NaNs are copied onto `targets`.
        public let mask: Int
        public let targets: [Int]
    }

    public struct Bounding: Codable, Sendable {
        public let type: String  // relu | hardtanh | fraction
        public let indices: [Int]
        public let min: Float?
        public let max: Float?
        /// For `fraction`: the column the clipped fraction is multiplied by.
        public let total: Int?
    }

    public let format: String
    public let version: Int
    public let model: String
    public let description: String
    public let dtype: String
    public let multiStep: Int
    public let timestepHours: Int
    public let numChannels: Int
    public let numDataNodes: Int
    public let numHiddenNodes: Int
    public let numEncoderEdges: Int
    public let encoderMaxInDegree: Int
    public let decoderInDegree: Int
    public let edgeDim: Int
    public let nodeAttrDim: Int
    public let mapperHeads: Int
    public let processor: Processor
    public let inputVars: [String]
    public let outputVars: [String]
    public let stateVars: [String]
    public let computedForcings: [String]
    public let prognosticInput: [Int]
    public let prognosticOutput: [Int]
    public let inputNormMul: [Float]
    public let inputNormAdd: [Float]
    public let outputNormMul: [Float]
    public let outputNormAdd: [Float]
    public let imputation: [Imputation]
    public let conditionalNan: ConditionalNaN
    public let boundings: [Bounding]
    public let accumulations: [String]

    public static func load(from url: URL) throws -> Manifest {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let manifest = try decoder.decode(Manifest.self, from: Data(contentsOf: url))
        guard manifest.format == format else {
            throw AIFSError.invalidBundle("\(url.lastPathComponent) is not an \(format) manifest")
        }
        guard manifest.version == supportedVersion else {
            throw AIFSError.invalidBundle(
                "bundle version \(manifest.version); this build reads version \(supportedVersion)")
        }
        return manifest
    }
}

public enum AIFSError: LocalizedError {
    case invalidBundle(String)
    case invalidInitialCondition(String)

    public var errorDescription: String? {
        switch self {
        case .invalidBundle(let why): "Invalid model bundle: \(why)"
        case .invalidInitialCondition(let why): "Invalid initial condition: \(why)"
        }
    }
}
