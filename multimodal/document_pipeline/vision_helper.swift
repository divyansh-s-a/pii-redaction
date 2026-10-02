// Apple Vision shim — objective, local, zero-install signals for the
// benchmark. Chosen because it needs no pip package and no network: the
// machine has no QR decoder, no OCR engine and no Pillow.
//
//   swift vision_helper.swift <image.png> <barcodes|text|faces|all>
//
// Emits one JSON object on stdout. Boxes are normalised [x0,y0,x1,y1] with
// the ORIGIN AT TOP-LEFT (Vision returns bottom-left; converted here) so they
// map straight onto PDF/pixel space without a second convention to remember.
//
// Known limit, deliberate: Vision's OCR covers 30 languages but NO Indic
// script. Devanagari/Kannada/Tamil/Telugu pages must go to a vision model.
import Foundation
import Vision
import CoreImage

struct Box: Codable { let x0: Double; let y0: Double; let x1: Double; let y1: Double }

func flip(_ r: CGRect) -> Box {           // Vision y-up -> image y-down
    Box(x0: Double(r.minX), y0: Double(1.0 - r.maxY),
        x1: Double(r.maxX), y1: Double(1.0 - r.minY))
}

// BATCH MODE. `swift vision_helper.swift` pays a compile-and-spawn on every
// invocation -- ~0.37s -- which dwarfs the detection itself. Face detection
// has to TILE a page (Vision will not find a passport photo inside a full A4
// scan), so a page costs 16 invocations and the 2043-page corpus cost 3.4
// hours, nearly all of it process startup. Passing every tile in one call
// makes it one spawn per page instead of sixteen.
//
//   swift vision_helper.swift <image> <mode>                  -> {...}
//   swift vision_helper.swift --batch <mode> <img1> <img2>... -> {"batch":[...]}
//
let args = CommandLine.arguments

func analyse(_ path: String, _ mode: String) -> [String: Any] {
    guard let img = CIImage(contentsOf: URL(fileURLWithPath: path)) else {
        return ["error": "cannot read image"]
    }
    let handler = VNImageRequestHandler(ciImage: img, options: [:])
    return run(handler, mode)
}

if args.count >= 3, args[1] == "--batch" {
    let mode = args[2]
    var results: [[String: Any]] = []
    for p in args.dropFirst(3) {
        var r = analyse(p, mode)
        r["path"] = p
        results.append(r)
    }
    let data = try! JSONSerialization.data(withJSONObject: ["batch": results])
    print(String(data: data, encoding: .utf8)!)
    exit(0)
}

guard args.count >= 2, let img = CIImage(contentsOf: URL(fileURLWithPath: args[1])) else {
    print("{\"error\":\"cannot read image\"}"); exit(1)
}
let mode = args.count >= 3 ? args[2] : "all"
let handler = VNImageRequestHandler(ciImage: img, options: [:])
let single = run(handler, mode)
let sdata = try! JSONSerialization.data(withJSONObject: single)
print(String(data: sdata, encoding: .utf8)!)
exit(0)

func run(_ handler: VNImageRequestHandler, _ mode: String) -> [String: Any] {
var out: [String: Any] = [:]
let enc = JSONEncoder()
func encodeBox(_ b: Box) -> [String: Double] {
    [ "x0": b.x0, "y0": b.y0, "x1": b.x1, "y1": b.y1 ]
}

if mode == "barcodes" || mode == "all" {
    let req = VNDetectBarcodesRequest()
    try? handler.perform([req])
    var found: [[String: Any]] = []
    for obs in (req.results ?? []) {
        var item: [String: Any] = [
            "symbology": obs.symbology.rawValue,
            "box": encodeBox(flip(obs.boundingBox)),
            "confidence": Double(obs.confidence),
        ]
        // payload is the objective ground truth: if a QR decodes, whatever it
        // contains is provably present on the page
        if let s = obs.payloadStringValue { item["payload"] = s }
        if #available(macOS 12.0, *), let d = obs.payloadData {
            item["payload_bytes"] = d.count
        }
        found.append(item)
    }
    out["barcodes"] = found
}

if mode == "faces" || mode == "all" {
    let req = VNDetectFaceRectanglesRequest()
    try? handler.perform([req])
    out["faces"] = (req.results ?? []).map { obs -> [String: Any] in
        ["box": encodeBox(flip(obs.boundingBox)),
         "confidence": Double(obs.confidence)]
    }
}

if mode == "text" || mode == "all" || mode == "text_upright" {
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    req.usesLanguageCorrection = true
    // "text_upright" pins revision 2, which never re-orients the image.
    // Revision 3 decided that dark ultrasound printouts were rotated 180
    // degrees and transcribed every glyph flipped (a 9-digit ID came back as
    // letter look-alikes) while keeping the boxes in place. readers.py asks for this
    // mode only when the tokens it got run right-to-left on the page.
    if mode == "text_upright" {
        req.revision = VNRecognizeTextRequestRevision2
    }
    try? handler.perform([req])
    var words: [[String: Any]] = []
    for obs in (req.results ?? []) {
        guard let top = obs.topCandidates(1).first else { continue }
        let line = top.string
        // Line-level boxes are too coarse to redact with: boxing a value inside
        // "a) Name: Priya Sharma" would also cover the label. Vision can return
        // the box for a SUBRANGE of the recognised string, so emit one entry
        // per whitespace-separated token with its own box. That restores the
        // word-index anchoring the born-digital path relies on.
        var tokens: [[String: Any]] = []
        var idx = line.startIndex
        while idx < line.endIndex {
            if line[idx] == " " { idx = line.index(after: idx); continue }
            var end = idx
            while end < line.endIndex && line[end] != " " {
                end = line.index(after: end)
            }
            let tok = String(line[idx..<end])
            if let b = try? top.boundingBox(for: idx..<end) {
                tokens.append(["text": tok,
                               "box": encodeBox(flip(b.boundingBox)),
                               "confidence": Double(top.confidence)])
            }
            idx = end
        }
        words.append(["text": line,
                      "box": encodeBox(flip(obs.boundingBox)),
                      "confidence": Double(top.confidence),
                      "tokens": tokens])
    }
    out["text"] = words
}

// LANGUAGE-AGNOSTIC TEXT REGIONS. VNDetectTextRectangles finds WHERE text is
// without trying to read it, so it reports Kannada, Devanagari and Tamil that
// VNRecognizeText cannot transcribe at all -- Vision supports no Indic script.
// Subtracting the regions the recogniser did account for leaves exactly the
// text no reader in this pipeline has ever seen.
if mode == "textregions" || mode == "all" {
    let req = VNDetectTextRectanglesRequest()
    req.reportCharacterBoxes = false
    try? handler.perform([req])
    var regions: [[String: Any]] = []
    for obs in (req.results as? [VNTextObservation]) ?? [] {
        let b = obs.boundingBox
        regions.append(["x0": b.minX, "x1": b.maxX,
                        "y0": 1 - b.maxY, "y1": 1 - b.minY,
                        "confidence": Double(obs.confidence)])
    }
    out["textregions"] = regions
}

return out
}
