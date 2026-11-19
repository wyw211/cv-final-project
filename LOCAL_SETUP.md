# Local setup and first experiment

Run PowerShell commands from `C:\Project\CV final project\ViSA`.
The virtual environment is in the parent directory (`..\.venv`).
The official source commit is `22ff906c28c63144d382c160d097b0de6f4f0510`.

## Environment

- Windows, Python 3.13.7
- NVIDIA RTX 5070 Ti, 16 GB VRAM; system RAM about 32 GB
- PyTorch 2.7.1+cu128 and torchvision 0.22.1+cu128
- CPU FAISS replaces the old `faiss-gpu` dependency. The default evaluation
  uses cosine distance without reranking; GPU FAISS reranking is not enabled.
- CUDA Toolkit installation is not needed for these prebuilt PyTorch wheels.

Recreate the environment (from the project parent directory):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r ViSA/requirements-windows.txt
```

`requirements-lock.txt` records the versions installed during initial verification.
Install locked non-PyTorch packages with PyPI and install the two PyTorch packages
from the CUDA index above; their `+cu128` builds are not hosted on normal PyPI.

## Assets

- `weights/jx_vit_base_p16_224-80ecf9dd.pth`
  - Source: https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
  - SHA256: `80ecf9dd5e3a58895e959af554c5666c4e7b4da4410de4f1f2b0025e93435d8c`
- CARGO official download: https://drive.google.com/file/d/1yDjyH0VtW7efxP3vgQjIqTx2oafCB67t/view

```powershell
..\.venv\Scripts\python.exe tools/download_cargo.py
```

Expected layout: `datasets/CARGO/{train,query,gallery}/Cam*/...jpg`.
The downloader preserves the official splits and verifies the ZIP CRCs.
Do not combine test images with training images.

## Verify before training

For checks without any training or parameter updates, use:

```powershell
..\.venv\Scripts\python.exe tools/verify_no_training.py
```

See `VERIFICATION.md` for results and the two evaluation-code fixes. This path
uses only training images for model inference, and synthetic fixtures for metrics.

The separate command below DOES perform two small training updates. Do not use
it when only read-only model verification is desired.

```powershell
..\.venv\Scripts\python.exe tools/smoke_test.py --steps 2 --batch-size 8
```

This verifies the weight hash, CUDA computation, real training images, all ViSA
losses, backward gradients, parameter updates, and inference features on the same
training images. It writes `output/smoke_test.json`; these are environment checks,
not Rank-1/mAP results. No benchmark evaluation or checkpoint selection occurs.

## Training preparation

The local config `configs/CARGO/visa_windows.yml` inherits the official model
architecture and losses. It sets the weight path, zero training loader workers,
training batch size 16, and evaluation batch size 32 for initial local bring-up.
The original config is unchanged. The CARGO filename parser uses `osp.basename`
so the same filenames work with both Windows and POSIX path separators.

Do not treat a run using batch size 16 as an exact reproduction of the published
training setup. Before the full baseline, reconcile the paper/supplement settings,
batch size, fixed seeds, all protocols, and memory requirements. Freeze a schedule
without using test metrics for model selection. Evaluate the final checkpoint only.
Use the same controlled setup for the proposed improvement.

LAGPeR is not included here. Obtain the official dataset from the instructor or
authors before preparing its experiments; do not substitute another dataset.

## Verification completed

- GPU tensor forward/backward: passed on RTX 5070 Ti (CUDA capability 12.0).
- ViSA model and DefaultTrainer imports and training CLI: passed.
- `pip check`: no broken requirements.
- Official CARGO ZIP: downloaded, CRC verified, and extracted.
- Training split: 51,451 images, 2,500 identities.
- Full model: 275,618,328 parameters with the 2,500-identity classifier.
- Real-image smoke test: batch size 16, two SGD/AMP steps, finite losses and
  gradients, parameter update verified, inference feature shape `[16, 1536]`.
- Peak allocated GPU memory in this smoke test: 5.265 GiB.
- This test uses a direct SGD loop with an initial AMP scale of 128. It does not
  validate a complete epoch, the DefaultTrainer optimizer/data pipeline, retrieval
  metrics, or a full reproduction. Full-run memory and runtime may differ.
- No full baseline training or test-set retrieval evaluation has been performed.

The next milestone is to verify the native training pipeline and reconcile the
paper's full-run settings before starting the controlled baseline experiment.

The CARGO paper/code settings audit is now in `CARGO_SETTINGS_AUDIT.md`. The new
`configs/CARGO/visa_official_single_gpu.yml` follows the README's batch-256 recipe;
the earlier Windows batch-16 file remains a bring-up profile. No training was
started during the audit. The upstream loss-coefficient and protocol ambiguities
are documented rather than silently changed.

Activation checkpointing is now available as an opt-in profile at
`configs/CARGO/visa_official_checkpointing.yml`. See `CHECKPOINTING.md` for exact
gradient equivalence and batch-256 capacity checks. The native training loops
also preserve contiguous gradients with `zero_grad(set_to_none=False)` for
PyTorch 2.x compatibility. No optimizer updates were performed in these checks.

A later user-authorized 50-step training check is documented in
`SHORT_TRAINING.md`: real optimizer updates, warmup timing, and fresh-trainer
checkpoint restore passed. AMP scaler state is now saved by DefaultTrainer.
The full 120-epoch baseline has not started.

Periodic latest/final saving is now configured and verified in
`CHECKPOINT_POLICY.md`: every five epochs, atomic writes, full optimizer/AMP/RNG
state, and final saving before evaluation. No test-metric checkpoint selection.
