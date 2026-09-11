# ACC-ViT-2: Hierarchical Selection of Elliptical-Galaxy Candidates

Companion catalog and inference implementation for **Hierarchical Image-based Selection of Elliptical Galaxy Candidates with ACC-ViT-2**.

ACC-ViT-2 uses a two-stage image classifier to select galaxies with smooth, round apparent morphologies from SDSS DR17 color images. Stage 1 separates ELL, CIG+EDG, and SPI. Stage 2 separates the ELL candidates into COM and INB; COM objects form the final candidate sample.

## Contents

| File | Description |
|---|---|
| `catalogs/beyond_hart16_com_candidates.xlsx` | 28,132 COM candidates selected from 138,625 Vavilova-preselected objects outside Hart16. |
| `MODEL.py` | ACC-ViT model definition. |
| `utils_p.py` | Model configuration and inference preprocessing. |
| `inference.py` | Checkpoint loading, stage routing, and prediction output. |
| `predict.py` | Command-line prediction entry point. |
| `requirements.txt` | Python dependencies for inference and catalog access. |
| `CATALOG_COLUMNS.md` | Column descriptions and units for the released catalog. |
| `WEIGHTS_SHA256.txt` | SHA-256 checksums for the two checkpoint files distributed through Releases. |

This package provides the beyond-Hart16 candidate catalog and inference code. It does not include the original image collection, training pipeline, baseline models, or experiment split files.

## Catalog

The Excel workbook contains a `Candidates` sheet and a `Column guide` sheet. The latter describes the columns and gives catalog summary counts. The same column descriptions are available in [CATALOG_COLUMNS.md](CATALOG_COLUMNS.md).

### Beyond-Hart16 sample

The 28,132 rows are COM candidates from the external application sample described in the paper. The catalog includes SDSS identifiers, coordinates, photometric and size information, ACC-ViT-2 scores, and comparison probabilities from the Vavilova catalog.

The model-score columns have the following meanings:

- `stage1_ell_probability`: Stage 1 softmax score for ELL.
- `stage2_com_probability`: Stage 2 softmax score for COM after routing to the ELL branch.
- `joint_confidence`: **the minimum of these two scores**, not their product and not a calibrated joint probability.
- `confidence_ge_0p90`: both scores are at least 0.90 (24,913 candidates).
- `confidence_ge_0p95`: both scores are at least 0.95 (23,858 candidates).

All 28,132 candidates are retained in the workbook; the confidence flags allow optional stricter selections. Apparent-size groups use R90 in arcseconds: Small, R90 < 4; Medium, 4 <= R90 < 6; Large, R90 >= 6.

Long SDSS identifiers are stored as text to preserve their digits. Preserve this type when reading or exporting the table. Blank values indicate unavailable measurements, not zeros. Probabilities from the comparison catalogs are distinct from ACC-ViT-2 scores.

Example:

```python
import pandas as pd

candidates = pd.read_excel(
    "catalogs/beyond_hart16_com_candidates.xlsx",
    sheet_name="Candidates",
    dtype={"sdss_bestobjid": "string"},
)
high_confidence = candidates[candidates["confidence_ge_0p95"] == True]
print(len(candidates), len(high_confidence))  # 28132, 23858
```

## Model weights

Download both checkpoint files from the **Releases** section of this repository and place them in a local `weights/` directory:

```text
weights/
  stage1_accvit.pth
  stage2_accvit.pth
```

These files are approximately 355 MiB each and are distributed as release attachments rather than ordinary Git files. They contain model parameters; use the accompanying model definition and configuration to load them. No separate initialization checkpoint is needed for inference.

Class indices are fixed:

| Stage | Index | Class |
|---|---|---|
| Stage 1 | 0 | ELL = COM + INB |
| Stage 1 | 1 | CIG+EDG |
| Stage 1 | 2 | SPI |
| Stage 2 | 0 | COM |
| Stage 2 | 1 | INB |

Only objects assigned to ELL by Stage 1 are passed to Stage 2. Final indices are `0_com`, `1_inb`, `2_cig_and_edg`, and `3_spi` (0–3 respectively).

## Run inference

From the repository directory, install the dependencies in your Python environment:

```bash
python -m pip install -r requirements.txt
```

A CPU inference smoke test was completed with the package versions recorded in `requirements-tested.txt`. This is a tested environment snapshot, not a guarantee of compatibility on every platform.

For GPU inference, use a PyTorch and torchvision installation compatible with your CUDA environment. CPU inference is also supported.

Place square SDSS RGB cutouts in a directory, such as `images/`. The original application uses 256 x 256 pixel JPEG cutouts at 0.396 arcsec/pixel. The script accepts JPEG and PNG files, including files in subdirectories. It converts images to RGB, resizes to 224 x 224 pixels, converts pixel values to tensors in [0, 1], and normalizes the channels using:

```text
mean = [0.046, 0.041, 0.030]
std  = [0.090, 0.075, 0.065]
```

These numbers apply to the RGB image channels. Raw FITS bands require the appropriate color-image construction before using this interface. Evaluation preprocessing does not apply random augmentation.

```bash
python predict.py --images images/ --output predictions.csv --device cpu
```

For a CUDA GPU:

```bash
python predict.py --images images/ --output predictions.csv --device cuda:0 --batch-size 8
```

Alternative checkpoint locations can be specified with `--stage1 PATH` and `--stage2 PATH`. Reduce the batch size if memory is limited.

The inference implementation sums the four branch logits with weights `[1, 1, 1, 1]`, then applies softmax separately at each stage. This follows the supplied application code. Averaging these logits instead would preserve argmax labels but change the confidence values.

## Prediction output

`predictions.csv` contains one row per input image:

| Column | Meaning |
|---|---|
| `filename` | Input image filename. |
| `object_id` | Filename without its extension; an SDSS identifier only if the image was named that way. |
| `full_path` | Local absolute path of the input image. |
| `first_stage_label`, `first_stage_category` | Stage 1 index and class name. |
| `first_stage_confidence` | Softmax score of the selected Stage 1 class. |
| `second_stage_confidence` | Softmax score of the selected Stage 2 class; blank for objects not routed to Stage 2. |
| `predicted_label`, `predicted_category` | Final index and class name. |

Select `predicted_category == "0_com"` to obtain COM candidates. For these rows, the first- and second-stage confidences correspond to the ELL and COM scores, respectively. Their minimum matches the definition of `joint_confidence`. The prediction CSV does not perform the catalog cross-matches needed to populate the photometric and comparison fields in the released workbook.

## Interpretation and citation

COM denotes an image-based morphology selection. It does not establish a physical elliptical-galaxy classification or exclude every face-on S0 galaxy or weak-disk system. Model scores are not calibrated estimates of sample purity. The beyond-Hart16 sample is subject to the Vavilova preselection described in the paper.

If using this catalog or these models, cite **Hierarchical Image-based Selection of Elliptical Galaxy Candidates with ACC-ViT-2** and acknowledge the relevant source catalogs described in the paper and workbook column guides.
