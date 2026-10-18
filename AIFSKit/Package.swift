// swift-tools-version: 6.0
// AIFS Single v2 forward pass and rollout in MLX, shared by the iPad app and a macOS CLI
// used to validate it against the PyTorch pipeline. Build with xcodebuild, not
// `swift build`: MLX's Metal shaders are only compiled by Xcode's build system.
import PackageDescription

let package = Package(
    name: "AIFSKit",
    platforms: [.iOS(.v18), .macOS(.v15)],
    products: [
        .library(name: "AIFSKit", targets: ["AIFSKit"]),
        .executable(name: "aifs-mlx", targets: ["aifs-mlx"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.32.3")),
    ],
    targets: [
        .target(
            name: "AIFSKit",
            dependencies: ["CAEC", .product(name: "MLX", package: "mlx-swift")]
        ),
        // libaec's CCSDS decoder, for GRIB2 open data (grid_ccsds). See Sources/CAEC/README.md.
        .target(
            name: "CAEC",
            exclude: ["LICENSE.txt", "README.md"],
            cSettings: [.headerSearchPath(".")]
        ),
        .executableTarget(name: "aifs-mlx", dependencies: ["AIFSKit"]),
    ]
)
