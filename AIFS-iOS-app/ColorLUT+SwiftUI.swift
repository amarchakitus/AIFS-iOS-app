import AIFSKit
import SwiftUI

extension ColorLUT {
    func swiftUIColor(_ p: Float) -> Color {
        let c = components(p)
        return Color(red: c.red, green: c.green, blue: c.blue)
    }
}
