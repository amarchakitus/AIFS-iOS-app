import SwiftUI

/// Licences and attribution: the models, the data they start from, and the open-source
/// software compiled into the app. Licence texts ship verbatim in `Licenses/`.
struct AttributionView: View {
    var body: some View {
        List {
            Section {
                VStack(alignment: .leading, spacing: 8) {
                    Text("AIFS Single v2").font(.headline)
                    Text("© European Centre for Medium-Range Weather Forecasts (ECMWF). The inference "
                        + "weights are licensed under CC BY 4.0. This app converts them to half precision "
                        + "for MLX and Core ML and runs them on this device.")
                        .font(.subheadline).foregroundStyle(.secondary)
                }
                Link("Model card on Hugging Face", destination: URL(string: "https://huggingface.co/ecmwf/aifs-single-2.0")!)
                Link("CC BY 4.0 licence", destination: URL(string: "https://creativecommons.org/licenses/by/4.0/")!)
                VStack(alignment: .leading, spacing: 8) {
                    Text("Aurora 1.5").font(.headline)
                    Text("© Microsoft Corporation. The model and its weights are licensed under the MIT licence. "
                        + "This app converts them to Core ML programs, with 8-bit weights in its Swin blocks, "
                        + "and runs them on this device's Neural Engine.")
                        .font(.subheadline).foregroundStyle(.secondary)
                }
                Link("Aurora 1.5 documentation", destination: URL(string: "https://microsoft.github.io/aurora/models.html#aurora-1-5-0-25")!)
            } header: {
                Text("Forecast models")
            } footer: {
                Text("Lang, S. et al. (2024). AIFS – ECMWF's data-driven forecasting system. arXiv:2406.01465.\n"
                    + "Aurora 1.5 is fine-tuned from Aurora: Bodnar, C. et al. (2025). A foundation model for the Earth system. "
                    + "Nature. doi:10.1038/s41586-025-09005-y.")
            }

            Section {
                VStack(alignment: .leading, spacing: 8) {
                    Text("ECMWF open data").font(.headline)
                    Text("Initial conditions contain modified ECMWF real-time open data, © ECMWF, used under "
                        + "CC BY 4.0 and the ECMWF Terms of Use, from the IFS and, for Aurora's cloud layers, AIFS. "
                        + "They are regridded to the model grid where needed and combined into the model's input "
                        + "state. Served by ECMWF, and mirrored on Amazon Web Services and Microsoft Azure.")
                        .font(.subheadline).foregroundStyle(.secondary)
                }
                Link("ECMWF open data", destination: URL(string: "https://www.ecmwf.int/en/forecasts/datasets/open-data")!)
                Link("ECMWF Terms of Use", destination: URL(string: "https://apps.ecmwf.int/datasets/licences/general/")!)
                VStack(alignment: .leading, spacing: 8) {
                    Text("Interpolation weights").font(.headline)
                    Text("From ECMWF's earthkit-geo, under the Apache License 2.0.")
                        .font(.subheadline).foregroundStyle(.secondary)
                }
                VStack(alignment: .leading, spacing: 8) {
                    Text("Place search").font(.headline)
                    Text("Search and place names by Apple Maps.")
                        .font(.subheadline).foregroundStyle(.secondary)
                }
            } header: {
                Text("Data")
            }

            Section {
                Text("Forecasts in this app are computed on this device. They are not produced, checked "
                    + "or endorsed by ECMWF or Microsoft, and should not be relied on where safety is at stake.")
                    .font(.subheadline).foregroundStyle(.secondary)
            } header: {
                Text("Not an official forecast")
            }

            Section {
                ForEach(Self.software, id: \.file) { item in
                    NavigationLink {
                        LicenseText(title: item.name, file: item.file)
                    } label: {
                        LabeledContent(item.name, value: item.licence)
                    }
                }
            } header: {
                Text("Open-source software")
            }
        }
        .navigationTitle("Licences & attribution")
        .navigationBarTitleDisplayMode(.inline)
    }

    private static let software: [(name: String, licence: String, file: String)] = [
        ("Aurora", "MIT", "aurora"),
        ("MLX Swift", "MIT", "mlx-swift"),
        ("MLX", "MIT", "mlx"),
        ("MLX C", "MIT", "mlx-c"),
        ("MLX third-party components", "Various", "mlx-third-party"),
        ("metal-cpp", "Apache 2.0", "metal-cpp"),
        ("{fmt}", "MIT", "fmt"),
        ("JSON for Modern C++", "MIT", "nlohmann-json"),
        ("Swift Numerics", "Apache 2.0", "swift-numerics"),
        ("libaec", "BSD 2-Clause", "libaec"),
    ]
}

/// A licence's full text, verbatim from the bundle.
private struct LicenseText: View {
    let title: String
    let file: String

    var body: some View {
        ScrollView {
            Text(text)
                .font(.footnote.monospaced())
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
        }
        .navigationTitle(title)
        .navigationBarTitleDisplayMode(.inline)
    }

    private var text: String {
        Bundle.main.url(forResource: file, withExtension: "txt")
            .flatMap { try? String(contentsOf: $0, encoding: .utf8) } ?? "Licence text missing from this build."
    }
}
