# AIFS on iPad

ECMWF's **AIFS Single v2** running entirely on an iPad: initial conditions downloaded from
ECMWF open data on the device, the forecast computed on the iPad's GPU with
[MLX](https://github.com/ml-explore/mlx-swift), and a map viewer for the result.

```
AIFS-iOS-app.xcodeproj   the iPad app project
AIFS-iOS-app/            app sources only (SwiftUI) -- everything here is built into the app
AIFSKit/                 Swift package: the model, rollout, open-data client, map rendering
  Sources/AIFSKit/         forward pass (AIFSModel), rollout (Forecast), forcings
  Sources/AIFSKit/OpenData GRIB2 decoder + ECMWF open-data client
  Sources/AIFSKit/Display  regridding and color scales for maps
  Sources/CAEC/            vendored libaec 1.1.7 decoder (CCSDS packing in open-data GRIB)
  Sources/aifs-mlx/        macOS CLI running the same code, for validation
aifs-mps/                the original PyTorch/MPS pipeline (see its README and CLAUDE.md),
                         plus the exporter that produces the iPad's model bundle
tools/make_icon.py       generates AIFS-iOS-app/AppIcon.icon (Liquid Glass, Icon Composer format)
```

## Why it is built this way

PyTorch does not run on iPadOS, so the network is re-implemented in MLX Swift and fed a
**bundle** exported from the checkpoint: fp16 weights, the encoder/decoder graphs
pre-arranged for gathers, and every constant anemoi applies around the network
(normalisation, imputation, bounding) resolved to plain numbers. See
`aifs-mps/aifs_mps/export.py`.

## Setup

```bash
cd aifs-mps
./aifs setup                       # Python runtimes + checkpoints (once)
./aifs export bundle               # -> exports/aifs-single-v2/ (541 MB)
cd ..
uv run tools/export_coreml.py aifs-mps/exports/aifs-single-v2   # + coreml/ (403 MB, optional)
```

The last step compiles the processor's dense layers and the decoder's dense second half into
Core ML models for the **Neural Engine** (`tools/export_coreml.py`). Without it the app runs
everything on the GPU; with it, the GPU keeps only the attention and graph gathers.

Open `AIFS-iOS-app.xcodeproj`, pick your team under Signing, run on the iPad. Then copy
the bundle into the app's Documents -- Finder → iPad → Files → AIFS, drag
`aifs-single-v2` in -- or from the command line:

```bash
cd aifs-mps/exports/aifs-single-v2
for f in $(find . -type f); do
  xcrun devicectl device copy to --device <UDID> --domain-type appDataContainer \
    --domain-identifier io.github.amarchakitus.aifs-mlx \
    --source "$f" --destination "Documents/aifs-single-v2/${f#./}"
done
```

In the app: **Download latest from ECMWF** (~125 MB, under a minute on good Wi-Fi),
pick a lead time, **Run forecast**, and watch steps appear in the viewer as they finish.

## Performance

Per 6-hour step on an iPad Pro 12.9" (M1, 16 GB):

| configuration | step | thermal state after 3–4 steps |
|---|---|---|
| everything on the GPU | 25 s cool, ~35 s once throttled | serious |
| + processor on the Neural Engine | 22.5 s | nominal |
| + decoder dense half on the Neural Engine | 17.9 s | nominal |
| + Neural Engine and GPU overlapped in the processor (default) | **~15.5 s** | nominal |

On an **iPhone 18 Pro Max** (A19 Pro, 12 GB) the same app runs at **~5.9 s per step** (48 h in
under a minute). It stays cool through a single run, but reached thermal state "serious"
after several back-to-back runs while charging. The phone's app memory limit (~6.4 GB)
is tighter than the iPad's, with ~1.2 GB left free mid-step.

On the iPad, a 48 h forecast takes about 2 minutes. The processor now runs as a wavefront: a layer's
attention for a chunk of the mesh needs only that chunk and its neighbours, so the GPU
computes attention for chunk i+1 while the Neural Engine already runs the next layer on
chunk i, and the processor costs little more than its attention. What made the difference, in order:
the Neural Engine runs the dense layers at ~6 TFLOPS against ~2.2 for the GPU, and does so
cool enough that the iPad no longer throttles; on the GPU, M1 matmuls only reach full speed
from ~17k rows, so the mappers run in 16 large chunks; and the mappers never build per-edge
1024-wide tensors (the edge embedding is folded algebraically into 11-dim terms).

The Neural Engine computes everything, layer norms included, in fp16, which costs accuracy
measurably but not meaningfully: 2 m temperature differs from PyTorch fp32 by 0.012 K RMS
at +6 h (GPU fp16: 0.0065 K) -- a tenth of the ~0.125 K by which two identical MPS runs
differ. Run with the GPU only by launching with `--no-ane`, or omit `coreml/` from the bundle.

## Validation

Every stage was checked against the PyTorch pipeline. Following the repo's rule, the model
was run once and every comparison was made against that cache.

| stage | compared with | agreement |
|---|---|---|
| forward pass, fp32 | anemoi `predict_step`, CPU fp32 | ≤ 3e-6 relative, all 120 outputs; NaN masks identical |
| forward pass, fp16 | same | 2t RMS 0.0064 K, MSLP 0.52 Pa: the same as PyTorch's own fp16 autocast (0.0059 K, 0.48 Pa) |
| input tensor + window roll | anemoi-inference runner, steps 0 and 1 | all 106 variables, ≤ 2.7e-7 |
| 3-step rollout | runner on MPS | fp16 noise only (2t RMS ≈ 0.008 K), not growing |
| open-data initial conditions | `opendata.build_input_state` | all 97 fields ≤ 3.3e-7 relative, NaN masks identical |
| Neural Engine path, on the iPad | MLX fp32, same initial state | 2t RMS 0.013 K, MSLP 0.87 Pa at +6 h; matches the Mac's Neural Engine |

To repeat the model checks on a Mac:

```bash
cd AIFSKit && xcodebuild -scheme aifs-mlx -destination 'platform=macOS' -configuration Release \
  -derivedDataPath .build/xcode build && cd ..
BIN=AIFSKit/.build/xcode/Build/Products/Release/aifs-mlx
$BIN --bundle aifs-mps/exports/aifs-single-v2 --fetch-ic latest --out /tmp/ic
$BIN --bundle aifs-mps/exports/aifs-single-v2 --ic /tmp/ic/ic_*.safetensors --steps 2 \
     --out /tmp/run --render /tmp/png
```

(`swift build` cannot compile MLX's Metal shaders; build with `xcodebuild`.)

## Limits

- **Single v2 only.** AIFS-ENS v2 needs a second exporter and model port (different
  anemoi version, noise injection, conditional layer norms).
- **Foreground only.** iPadOS stops GPU work in the background, so the app keeps the
  screen awake while a forecast runs.
- A free Apple developer account re-signs every 7 days; reinstall from Xcode when the app
  stops launching.
