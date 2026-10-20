// swift-tools-version: 6.0
// AIFS Single v2 forward pass and rollout in MLX, shared by the iPad app and a macOS CLI
// used to validate it against the PyTorch pipeline. Build with xcodebuild, not
// `swift build`: MLX's Metal shaders are only compiled by Xcode's build system.
import PackageDescription

let package = Package(
    name: "AIFSKit",
    platforms: [.iOS("27.0"), .macOS("27.0")],
    products: [
        .library(name: "AIFSKit", targets: ["AIFSKit"]),
        .executable(name: "aifs-mlx", targets: ["aifs-mlx"]),
        .executable(name: "aurora-ane", targets: ["aurora-ane"]),
        .executable(name: "graphcast-ane", targets: ["graphcast-ane"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.32.3")),
    ],
    targets: [
        .target(
            name: "AIFSKit",
            dependencies: ["CAEC", "AuroraANE", "GraphCastANE", .product(name: "MLX", package: "mlx-swift")]
        ),
        // Aurora 1.5 on the Neural Engine (Core ML and Accelerate only; no MLX), from the
        // programs aurora-1p5/aurora_1p5/ane_export.py writes.
        .target(
            name: "AuroraANE",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(name: "aurora-ane", dependencies: ["AuroraANE"]),
        // GraphCast operational on the Neural Engine, from the programs
        // graphcast-ane/graphcast_ane/ane_export.py writes (uses AuroraANE's safetensors reader).
        .target(
            name: "GraphCastANE",
            dependencies: ["AuroraANE"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(name: "graphcast-ane", dependencies: ["GraphCastANE"]),
        // libaec's CCSDS decoder, for GRIB2 open data (grid_ccsds). See Sources/CAEC/README.md.
        .target(
            name: "CAEC",
            exclude: ["LICENSE.txt", "README.md"],
            cSettings: [.headerSearchPath(".")]
        ),
        .executableTarget(name: "aifs-mlx", dependencies: ["AIFSKit"]),
    ]
)
