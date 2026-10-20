// GraphCast on the Neural Engine from the command line (macOS): the bundle's self-test.
//
//   graphcast-ane --bundle DIR [--steps N] [--forecast]
import Foundation
import GraphCastANE

var bundle: String?
var steps = 1
var forecast = false
var args = CommandLine.arguments.dropFirst()
while let a = args.popFirst() {
    switch a {
    case "--bundle": bundle = args.popFirst()
    case "--steps": steps = Int(args.popFirst() ?? "") ?? 1
    case "--forecast": forecast = true
    default:
        FileHandle.standardError.write("unknown argument \(a)\n".data(using: .utf8)!)
        exit(2)
    }
}
guard let bundle else {
    FileHandle.standardError.write("usage: graphcast-ane --bundle DIR [--steps N]\n".data(using: .utf8)!)
    exit(2)
}
do {
    let log: (String) -> Void = { print($0); fflush(stdout) }
    if forecast {
        try GraphCastSelfTest.forecast(directory: URL(filePath: bundle), log: log)
    } else {
        try GraphCastSelfTest.run(directory: URL(filePath: bundle), steps: steps, log: log)
    }
} catch {
    print("failed: \(error)")
    exit(1)
}
