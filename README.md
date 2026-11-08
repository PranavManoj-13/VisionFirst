# VisionFirst

Android edge intelligence project for DSC4109: Edge Intelligence and Automation, by Maitreya Ganu, Pranav M, and Neelay S Kamat.

VisionFirst grades retinal fundus photographs using a trained MobileNetV3-Large ordinal model running locally with LiteRT. The app includes camera capture, image selection, results, About, and Architecture pages. It uses a white, orange, and black theme, system sans-serif typography, and project credits in the hamburger menu. No gradients, uploads, accounts, analytics, or network permission are used.

## Workspace

```text
android-app/                       Android source, build configuration, Gradle wrapper
  app/src/main/assets/             Bundled LiteRT model and export metadata
  app/src/androidTest/             Device parity tests and golden fixtures
data/
  train.csv                        Original labels
  colored_images/                  All 3,662 original PNG images
dr_model/
  model.py                         MobileNetV3-Large and ordered threshold head
  data.py                          Data audit, grouped splits, preprocessing
  train.py                         Training, calibration, held-out evaluation
  metrics.py                       QWK, accuracy, F1, AUC, screening metrics
  export.py                        LiteRT/TorchScript export and parity checks
  predict.py                       Checkpoint reference inference
  plot_evaluation.py               PPT-ready evaluation figures
  paths.py                         Default data and output locations
requirements/
  training.txt                     Pinned training/reference dependencies
  export.txt                       Linux converter dependencies
tests/                             Python tests for the retained model pipeline
outputs/
  deployment.json                  Links and hashes for retained deployment artifacts
  runs/20261108T074317_338859Z/       Original model run, preserved intact
    checkpoints/best.pt            Trained and calibrated checkpoint
    metadata/                      Configuration, audit, exact split manifest
    logs/                          Epoch history
    metrics/                       Calibration and held-out test metrics
    predictions/                   Per-image test probabilities and grades
  exports/deployed-model/
    models/retinascan.tflite        Exported model matching the Android asset
    metadata/                      Model contract and converter environment
    fixtures/                      Python input/output parity references
  evaluation/deployed-model/
    png/                           Eight 300-DPI figures for PPT
    svg/                           Matching vector figures
    tables/                        Metrics JSON and per-class CSV
    captions.txt                   Figure captions and evaluation scope
    provenance.json                Model hashes and plot-generation provenance
  apk/VisionFirst-debug.apk        Latest installable debug APK (footer removed)
  app-verification/device-tests.txt Recorded on-device verification results
```

Abandoned experiments, failed runs, rejected-rule diagnostics, temporary Colab handoffs, redundant archives, and regenerable Android build/cache files have been removed. The source dataset, original trained model, measured results, exported model, evaluation figures, app, and verification tests are retained. Local IDE settings remain available; generated files are ignored by Git.

## Model and measured results

The deployed checkpoint was selected at epoch 5 using validation QWK. Training finished at epoch 10 after five epochs without improvement. It uses the original ordered head and threshold-count prediction rule.

| Held-out test metric | Value |
|---|---:|
| Images | 723 |
| Exact-grade accuracy | 74.41% |
| Quadratic weighted kappa | 0.8840 |
| Sensitivity, grade ≥ 2 | 93.03% |
| Specificity, grade ≥ 2 | 87.84% |

These are saved PyTorch checkpoint test results for the model exported to Android. Export/device parity checks establish numerical agreement on fixtures, not a separate full-test Android evaluation. No clinical validation is claimed.

The torchvision MobileNetV3-Large backbone starts from ImageNet weights. Global average pooling produces 960 features, projected to 1,280 features with hard-swish and training dropout. A shared severity score and four strictly increasing cut points produce ordered logits. Training uses ordinal binary cross-entropy, AdamW, and cosine learning-rate decay. Positive temperature calibration is fitted on validation data.

Mobile input is float32 RGB `[1,224,224,3]` with raw values in [0,255]. Decode EXIF orientation, convert to sRGB/RGB, and bilinearly resize to a square. Division by 255 and ImageNet normalization are embedded in the model: **do not normalize again in Android**. The four outputs are `P(grade > k)`, k=0…3. Count values strictly greater than 0.5 for grade 0…4. Five class probabilities are `[1-p0, p0-p1, p1-p2, p2-p3, p3]`.

The Android implementation uses a Pillow-compatible antialiased bilinear resize and LiteRT 1.4.2 CPU Interpreter. It checks model SHA256 and tensor shapes at runtime. Inference failures produce an error, never a fixed example. Images above 24 megapixels are rejected to bound memory. Camera files use private temporary cache storage and are removed after loading/cancellation. Session images are held in memory and reset on process termination.

## Data isolation

Every training run audits decoded pixels before splitting. Pixel-identical images stay in one partition. Thirty conflicting-label groups containing 62 images are excluded without modifying source images. Ninety-three same-label duplicate groups remain grouped. Default retained splits: 2,159 training / 718 validation / 723 test images.

Splits are approximately 60/20/20 by independent groups within each grade, with seed 42. Classes require at least five groups. Patient IDs are unavailable, so exact-pixel grouping does not exclude patient overlap or near duplicates. Use validation for development; do not repeatedly choose models using observed test scores.

## Python environment and training

Run commands from the repository root. Use the existing Conda environment:

```sh
conda activate retinascan
```

For a fresh environment:

```sh
conda create -n retinascan python=3.11 pip -y
conda activate retinascan
python -m pip install -r requirements/training.txt
```

Audit or train a **new** run without overwriting the retained deployment:

```sh
python -m dr_model.train --validate-only
python -m dr_model.train --device mps --batch-size 16 --workers 0
```

Outputs go to `outputs/runs/<UTC timestamp>/`. `--output` can specify a new directory; existing directories are rejected. `--data-root` defaults to this repository's `data/`. Automatic device selection prefers CUDA, then MPS, then CPU. Lower batch size to 8 for memory constraints. Initial pretrained weights download if not cached. Training saves the best validation-QWK checkpoint, calibrates it on validation, and evaluates test at completion. New runs do not automatically replace the app asset.

Reference inference:

```sh
python -m dr_model.predict outputs/runs/20261108T074317_338859Z/checkpoints/best.pt data/colored_images/Moderate/IMAGE_ID.png
```

## Evaluation figures for PPT

Existing figures are in `outputs/evaluation/deployed-model/`. To regenerate into a new timestamped folder:

```sh
conda activate retinascan
conda install -c conda-forge matplotlib=3.9.4
python -m dr_model.plot_evaluation outputs/runs/20261108T074317_338859Z
```

Eight plots: metric summary, count/percentage confusion matrices, per-class precision/recall/F1, ordinal ROC curves, training history, grade-error distribution, and split composition. Output includes 300-DPI PNG, vector SVG, tables, captions, and provenance. The script checks saved predictions against metrics and verifies that the run matches the model bundled in the app. It does not retrain or rerun inference.

## Reproduce mobile export

The final export is retained in `outputs/exports/deployed-model/`. For a fresh conversion, use Linux/Colab and a separate Python 3.11 environment with `requirements/export.txt`; the LiteRT Torch converter resolves its own compatible dependencies. Copy the `dr_model/` package, checkpoint, export requirements, and real parity images.

```sh
python -m pip install -r requirements/export.txt
python -m dr_model.export outputs/runs/20261108T074317_338859Z/checkpoints/best.pt --format tflite --images image0.png image1.png image2.png
```

New bundles go to `outputs/exports/<UTC timestamp>/`, with `models/`, `fixtures/`, and `metadata/`. Checks cover output probability tolerance, grade agreement, ordering, and tensor shapes/types. Metadata is written only after successful checks. The model is float32; quantization is not part of this deployment. Host conversion timings are not phone benchmarks.

For a local reference-only TorchScript export, use `--format torchscript`. Android uses the `.tflite` artifact.

## Android app and APK

Open `android-app/` in Android Studio. Use JDK 17 or 21 and Android SDK 35. Run on Android 9/API 28 or newer. Gradle dependencies download on first sync; the installed app runs offline.

```sh
cd android-app
./gradlew :app:assembleDebug
```

Gradle recreates build/cache folders as needed. Newly built APK: `android-app/app/build/outputs/apk/debug/app-debug.apk`. The retained installable copy is `outputs/apk/VisionFirst-debug.apk`; replace that copy manually when intentionally distributing a newer build.

For release signing, choose **Build → Generate Signed Bundle / APK → APK**, supply a keystore, and select the release variant. Keep the keystore secure for future updates. The retained APK is debug-signed; no release-signed APK has been generated.

The package/application ID remains `com.pranav.retinascan`; the visible app name is **VisionFirst**. The header and hamburger menu provide Retinal scan, About, and Architecture pages. Credits are in the menu; there is no persistent footer.

## Verification

```sh
python -m unittest discover -s tests -v
```

Python tests cover data isolation, conflicting labels, RGB handling, metrics, calibration, ordered outputs, full backward passes, and TorchScript roundtrips. MPS tests skip if the execution environment cannot access the GPU.

From `android-app/`, with a connected device:

```sh
./gradlew :app:connectedDebugAndroidTest
```

Recorded device checks passed for all five exported inputs (probability tolerance 1e-4, identical grades) and both exact Pillow resize fixtures (upsampling/downsampling). Raw fixture images are in the test APK only. Production assets contain the model and metadata, not retinal test images.

Camera capture/cancellation, image upload, navigation/rotation, large text, and the final visual layout should be checked manually before distribution. Fundus imaging requires appropriate retinal equipment; a normal photo of the outside of an eye is not suitable. This educational prototype does not replace professional eye assessment.
