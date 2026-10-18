import SwiftUI

/// The privacy policy, in the app as the App Store asks. Kept in step with
/// `docs/privacy.md`, the copy hosted for the App Store Connect listing.
struct PrivacyPolicyView: View {
    private let sections: [(title: String, body: String)] = [
        ("In short", "AI Forecast Factory does not collect, store or share personal data. There are no accounts, "
            + "no analytics, no advertising and no tracking."),
        ("Forecasts", "The weather model runs entirely on your iPhone or iPad; nothing about your forecasts leaves it."),
        ("Your location", "Only if you choose \u{201C}Use my current location\u{201D} in Places: your location is used on "
            + "the device to find the nearest model grid point, and sent to Apple's MapKit service to look up the "
            + "place's name. It is kept only as that saved place, on your device."),
        ("Place search", "What you type when adding a place is sent to Apple's MapKit service to find matching "
            + "places. Apple's privacy policy applies to that service."),
        ("Downloads", "The app downloads weather data from ECMWF and its mirrors on Amazon Web Services and "
            + "Microsoft Azure, and the forecast model from Hugging Face. These requests carry nothing about you "
            + "beyond what any web request does, such as your IP address, which those services handle under their "
            + "own privacy policies."),
        ("What stays on your device", "Saved places, settings, downloaded data and forecasts are kept in the app's "
            + "own storage and are removed when you delete the app."),
    ]

    var body: some View {
        List {
            ForEach(sections, id: \.title) { section in
                Section(section.title) {
                    Text(section.body).font(.subheadline)
                }
            }
            Section {
                Text("Last updated 17 October 2026").font(.footnote).foregroundStyle(.secondary)
            }
        }
        .navigationTitle("Privacy policy")
        .navigationBarTitleDisplayMode(.inline)
    }
}
