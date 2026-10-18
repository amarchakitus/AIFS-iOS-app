import SwiftUI

/// The model as a tiny network: input nodes on the left (encoder), the hidden mesh in the
/// middle (processor), output nodes on the right (decoder). The part running now glows;
/// the rest stays dim. While nothing is running yet (loading), the whole thing breathes.
struct NetworkGlyph: View {
    enum Stage { case encoder, processor, decoder }

    /// Nil while loading or between phases.
    let stage: Stage?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// The engine's phase names ("Encoder", "Processor", "Decoder", "Loading model").
    init(phase: String) {
        switch phase.lowercased() {
        case "encoder": stage = .encoder
        case "processor": stage = .processor
        case "decoder": stage = .decoder
        default: stage = nil
        }
    }

    /// Room for the glow. A Canvas clips to its frame, and the glow reaches ~6 pt past
    /// the outer nodes; the canvas is this much larger all round and pulled back in with
    /// negative padding, so the glyph's footprint stays 58 x 22.
    private static let bleed: CGFloat = 7

    var body: some View {
        TimelineView(.animation(paused: reduceMotion)) { timeline in
            let t = timeline.date.timeIntervalSinceReferenceDate
            let pulse = reduceMotion ? 1 : 0.7 + 0.3 * (0.5 + 0.5 * sin(t * 4))
            Canvas { context, size in
                let b = Self.bleed
                draw(in: &context, origin: CGPoint(x: b, y: b),
                     size: CGSize(width: size.width - 2 * b, height: size.height - 2 * b), pulse: pulse)
            }
        }
        .frame(width: 58 + 2 * Self.bleed, height: 22 + 2 * Self.bleed)
        .padding(-Self.bleed)
        .accessibilityElement()
        .accessibilityLabel(stage.map { "\($0)".capitalized } ?? "Loading")
    }

    private func draw(in context: inout GraphicsContext, origin: CGPoint, size: CGSize, pulse: Double) {
        func at(_ x: Double, _ y: Double) -> CGPoint {
            CGPoint(x: origin.x + x * size.width, y: origin.y + y * size.height)
        }
        let inputs = [0.18, 0.5, 0.82].map { at(0.06, $0) }
        let mesh = [at(0.37, 0.3), at(0.37, 0.7), at(0.63, 0.3), at(0.63, 0.7)]
        let outputs = [0.18, 0.5, 0.82].map { at(0.94, $0) }

        // Each part: its nodes, and the edges that carry data into it.
        let encoderEdges = inputs.flatMap { a in mesh[0 ..< 2].map { (a, $0) } }
        let processorEdges = [(mesh[0], mesh[2]), (mesh[1], mesh[3]), (mesh[0], mesh[3]),
                              (mesh[1], mesh[2]), (mesh[0], mesh[1]), (mesh[2], mesh[3])]
        let decoderEdges = mesh[2 ..< 4].flatMap { a in outputs.map { (a, $0) } }
        let parts: [(Stage, [CGPoint], [(CGPoint, CGPoint)])] = [
            (.encoder, inputs, encoderEdges),
            (.processor, mesh, processorEdges),
            (.decoder, outputs, decoderEdges),
        ]

        let dim = Color.secondary.opacity(stage == nil ? 0.25 + 0.3 * pulse : 0.3)
        let lit = Color.accentColor.opacity(pulse)

        func drawPart(_ ctx: inout GraphicsContext, nodes: [CGPoint], edges: [(CGPoint, CGPoint)], color: Color, radius: Double) {
            var lines = Path()
            for (a, b) in edges { lines.move(to: a); lines.addLine(to: b) }
            ctx.stroke(lines, with: .color(color.opacity(0.7)), lineWidth: 0.8)
            for p in nodes {
                ctx.fill(Path(ellipseIn: CGRect(x: p.x - radius, y: p.y - radius, width: radius * 2, height: radius * 2)),
                         with: .color(color))
            }
        }

        // Dim parts first, then the active one on top with a glow.
        for (part, nodes, edges) in parts where part != stage {
            drawPart(&context, nodes: nodes, edges: edges, color: dim, radius: 2.1)
        }
        if let (_, nodes, edges) = parts.first(where: { $0.0 == stage }) {
            context.drawLayer { layer in
                layer.addFilter(.shadow(color: Color.accentColor.opacity(0.9 * pulse), radius: 3.5))
                drawPart(&layer, nodes: nodes, edges: edges, color: lit, radius: 2.5)
            }
        }
    }
}
