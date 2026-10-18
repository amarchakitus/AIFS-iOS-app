import SwiftUI

/// Licences and attribution: the model, the data it starts from, and the open-source
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
            } header: {
                Text("Forecast model")
            } footer: {
                Text("Lang, S. et al. (2024). AIFS – ECMWF's data-driven forecasting system. arXiv:2406.01465.")
            }

            Section {
                VStack(alignment: .leading, spacing: 8) {
                    Text("ECMWF open data").font(.headline)
                    Text("Initial conditions contain modified ECMWF real-time open data, © ECMWF, used under "
                        + "CC BY 4.0 and the ECMWF Terms of Use. They are regridded to the model grid and "
                        + "combined into the model's input state. Served by ECMWF, and mirrored on Amazon "
                        + "Web Services and Microsoft Azure.")
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
                    + "or endorsed by ECMWF, and should not be relied on where safety is at stake.")
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
