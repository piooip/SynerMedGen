# SynerMedGen

**Synergizing Medical Multimodal Understanding with Generation via Task Alignment**

Weiren Zhao, Yi Dong, Cheng Chen · The University of Hong Kong · ICML 2026

[Code](https://github.com/piooip/SynerMedGen) · [SynerMed dataset](https://huggingface.co/datasets/tuy55/SynerMed) · [BAGEL base model](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT)

SynerMedGen studies generation-aligned understanding for medical image synthesis, building on BAGEL. This guide covers the source code, dataset setup, fine-tuning, and batch image generation.

## Release status

The source package contains the model implementation, training entry point, medical editing inference entry point, and dataset loaders. Fine-tuned SynerMedGen weights are not bundled; download the BAGEL initialization below or supply your own trained checkpoint.

As of the release audit on **2026-09-06**, the dataset repository was reachable but `SynerMed.zip` was not yet listed on its main branch. The download command below becomes usable once the archive upload is committed. The public code repository still contained the project page; the commands require a checkout containing `train/`, `data/`, `modeling/`, and `eval/`.

The training example below describes the supplied mixed understanding/generation configuration. Separate Stage I/Stage II recipes, the inter-stage weight-transfer procedure, and the exact medical evaluation protocol are not yet included, so this example alone is not a complete reproduction of the paper.

## Installation

Use Linux, Python 3.10, and NVIDIA GPUs compatible with the installed CUDA/PyTorch and FlashAttention builds. The training example uses eight GPUs. Inference loads a complete model on each GPU; increasing the process count distributes images, not model parameters.

```bash
git clone https://github.com/piooip/SynerMedGen.git
cd SynerMedGen

conda create -n synermed python=3.10 -y
conda activate synermed

python -m pip install -r requirements.txt
python -m pip install pandas==2.3.3 pyarrow==21.0.0 huggingface_hub==0.35.3
python -m pip install flash_attn==2.5.8 --no-build-isolation

export PYTHONPATH="$PWD"
```

Run all subsequent commands from this repository root. If you already have the source in a directory named `Synermed`, enter that directory instead of cloning.

The supplemental installation command adds pandas, which the Parquet loader requires, and aligns PyArrow/Hugging Face Hub with the inspected development environment. `requirements.txt` still contains older pins for those two packages. FlashAttention is required by the model even though its requirements entry is commented out. Building it requires a compatible CUDA toolkit and compiler.

The inspected development environment uses Python 3.10.18, PyTorch 2.5.1, torchvision 0.20.1, Transformers 4.49.0, and FlashAttention 2.5.8. This is an environment reference, not a guarantee for other CUDA/driver combinations.

Check imports before starting a large job:

```bash
python -c "import torch, flash_attn, pyarrow, pandas; print('CUDA available:', torch.cuda.is_available())"
python eval/gen/gen_images_mp_imgedit.py --help
```

## Prepare the dataset

Download the archive from [tuy55/SynerMed](https://huggingface.co/datasets/tuy55/SynerMed), once it appears in **Files and versions**:

```bash
hf download tuy55/SynerMed SynerMed.zip --repo-type dataset --local-dir ./datasets
unzip ./datasets/SynerMed.zip -d ./datasets
export SYNERMED_DATA_ROOT="$PWD/datasets/SynerMed"
```

The inspected archive is approximately 102 GiB; its extracted contents occupy approximately 109 GiB. Allow at least 230 GiB for the archive and extracted data, plus additional space for model weights and training outputs.

```text
datasets/SynerMed/
├── SynerMed.parquet       # Generation data with embedded image bytes
├── SynerMed.jsonl         # Understanding conversations
└── slice/
    ├── t1/
    ├── t1ce/
    ├── t2/
    ├── flair/
    ├── ct_FDG/
    ├── pet_FDG/
    └── ...               # Other CT/MR/CBCT image directories
```

The Parquet file contains 2,506,240 rows in 612 row groups. Its generation fields are `image_list` (a list of encoded image bytes) and `instruction_list` (a list of lists of instruction strings). The edit loader samples a source/target transition from each row; the row count is not a count of unique patients or images.

The JSONL loader expects `image` and `conversations` fields. Image paths are relative to `slice/`; conversations use `from: "human"` / `from: "gpt"` and `<image>` markers. Use the supplied annotations directly; no PNG-to-Parquet conversion is needed.

### Configure local paths and the Parquet index

The checked-in `data/dataset_info.py` still points to development-machine paths. Add the following imports and replace its `DATASET_INFO` dictionary with this configuration; retain its existing loader imports and `DATASET_REGISTRY`:

```python
import os
from pathlib import Path

DATA_ROOT = Path(os.environ["SYNERMED_DATA_ROOT"]).expanduser().resolve()
REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_INFO = {
    "unified_edit": {
        "seedxedit_multi": {
            "data_dir": str(DATA_ROOT),
            "parquet_info_path": str(
                REPO_ROOT / "data_bagel/Synermed/seedxedit_multi_nas.json"
            ),
        },
    },
    "vlm_sft": {
        "llava_ov": {
            "data_dir": str(DATA_ROOT / "slice"),
            "jsonl_path": str(DATA_ROOT / "SynerMed.jsonl"),
        },
    },
}
```

Regenerate the index for your local path. Its keys must match the full paths returned by the loader; the supplied bare key `"SynerMed.parquet"` does not match an absolute `data_dir`.

```bash
python - <<'PY'
import json
import os
from pathlib import Path
import pyarrow.parquet as pq

root = Path(os.environ["SYNERMED_DATA_ROOT"]).expanduser().resolve()
parquet = root / "SynerMed.parquet"
metadata = pq.ParquetFile(parquet).metadata
index = {
    str(parquet): {
        "num_row_groups": metadata.num_row_groups,
        "num_rows": metadata.num_rows,
    }
}
output = Path("data_bagel/Synermed/seedxedit_multi_nas.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(index, indent=2) + "\n")
print(f"Indexed {metadata.num_rows} rows in {metadata.num_row_groups} row groups")
PY
```

### Configure the training mixture

Create `data/configs/synermed.yaml` with:

```yaml
unified_edit:
  dataset_names: [seedxedit_multi]
  image_transform_args:
    image_stride: 16
    max_image_size: 1024
    min_image_size: 512
  vit_image_transform_args:
    image_stride: 14
    max_image_size: 518
    min_image_size: 224
  is_mandatory: true
  num_used_data: [1]
  weight: 1

vlm_sft:
  dataset_names: [llava_ov]
  image_transform_args:
    image_stride: 14
    max_image_size: 980
    min_image_size: 378
    max_pixels: 2007040
  frame_sampler_args:
    max_num_frames: 12
    min_num_frames: 8
  is_mandatory: false
  shuffle_lines: true
  shuffle_seed: 0
  num_used_data: [2560000]
  weight: 1
```

`num_used_data` has different meanings in these loaders: for `unified_edit` it is the number of **Parquet file selections**, while for `vlm_sft` it is the maximum number of JSONL records to use. With one Parquet file, use `[1]`; its row groups supply the distributed work units. Using `[2560000]` for the edit group repeats the file millions of times.

Remove unused dataset groups entirely. In particular, do not retain the `t2i_pretrain` block from `example.yaml`: setting its weight to zero does not prevent its directory from being opened during initialization.

The JSONL loader reads the entire annotation file into memory on each rank before taking its subset. Plan sufficient host RAM for the roughly 7 GiB JSONL file and Python object overhead per process. For a small smoke run, use a separate small JSONL file and update `jsonl_path`; reducing `num_used_data` alone does not avoid the initial full-file read.

## Prepare model weights

Download the [BAGEL initialization](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT):

```bash
hf download ByteDance-Seed/BAGEL-7B-MoT --local-dir ./models/BAGEL-7B-MoT
export BASE_MODEL_DIR="$PWD/models/BAGEL-7B-MoT"
```

Keep the model configuration, tokenizer files, and VAE with the weights. The supplied training and inference paths use:

```text
models/BAGEL-7B-MoT/
├── ema.safetensors
├── ae.safetensors
├── llm_config.json
├── vit_config.json
├── tokenizer_config.json
├── tokenizer.json
├── vocab.json
└── merges.txt
```

BAGEL weights provide an initialization and a baseline; they are not the medical fine-tuned weights used for the paper's results.

## Training

After completing the dataset configuration above, start single-node, eight-GPU fine-tuning:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  train/pretrain_unified_navit.py \
  --dataset_config_file data/configs/synermed.yaml \
  --model_path "$BASE_MODEL_DIR" \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume_from "$BASE_MODEL_DIR" \
  --finetune_from_hf True \
  --auto_resume False \
  --resume_model_only True \
  --finetune_from_ema True \
  --results_dir results/synermed \
  --checkpoint_dir results/synermed/checkpoints \
  --wandb_project synermedgen \
  --wandb_name synermed \
  --wandb_offline True \
  --log_every 1 \
  --lr 2e-5 \
  --num_workers 1 \
  --num_replicate 1 \
  --num_shard 8 \
  --expected_num_tokens 10240 \
  --max_num_tokens 11520 \
  --max_num_tokens_per_sample 10240
```

This keeps the main hyperparameters of the supplied training command, while using local dataset paths, explicit output directories, and offline W&B logging. The default training limit is `total_steps=500000`; set `--total_steps` explicitly for your run. This default is not a paper-specific training schedule.

`max_latent_size=64` matches the supplied model setup. The token-budget arguments control packed sequence sizes rather than a conventional fixed image batch size. For fewer GPUs, change `CUDA_VISIBLE_DEVICES`, `--nproc_per_node`, and `--num_shard` together; this does not guarantee that the model will fit in the available memory.

### Checkpoints and restarting

The current implementation saves `model.safetensors` under step-numbered directories such as `results/synermed/checkpoints/0000200/`. Periodic saving is hard-coded to every 200 steps; `--save_every` does not change it, and stopping between save points does not trigger an extra final save.

EMA, optimizer, scheduler, and dataset-state saves are commented out in `train/fsdp_utils.py`. Consequently, the example sets `--auto_resume False`: automatic resume would try to restore missing optimizer/scheduler files.

To restart from saved model weights, rerun the training command with these replacements:

```bash
--resume_from /absolute/path/to/results/synermed/checkpoints/0000200 \
--auto_resume False \
--resume_model_only True \
--finetune_from_ema False
```

Keep `--model_path "$BASE_MODEL_DIR"` for the configurations/tokenizer/VAE, and use a new results/checkpoint directory for the restarted run. This is a weights-only restart: optimizer, learning-rate schedule, step counter, and data position start again.

## Inference

The release entry point is `eval/gen/gen_images_mp_imgedit.py`. It contains the same executable code as the development script named `gen_images_mp_imgedit1.py`.

### Prepare an inference model directory

For a BAGEL baseline, set:

```bash
export INFER_MODEL_DIR="$BASE_MODEL_DIR"
```

For medical inference, use a complete fine-tuned model directory containing the files listed above. If you only have `model.safetensors` from the current training code, assemble a separate inference directory as follows. Replace the checkpoint path with one that exists:

```bash
CKPT_DIR="/absolute/path/to/results/synermed/checkpoints/0015200"
INFER_MODEL_DIR=$(mktemp -d "$PWD/models/SynerMedGen-inference.XXXXXX")

for file in ae.safetensors llm_config.json vit_config.json \
            tokenizer_config.json tokenizer.json vocab.json merges.txt; do
  ln -s "$BASE_MODEL_DIR/$file" "$INFER_MODEL_DIR/$file"
done
ln -s "$CKPT_DIR/model.safetensors" "$INFER_MODEL_DIR/ema.safetensors"
export INFER_MODEL_DIR
```

The inference script currently requires the filename `ema.safetensors`. This symlink lets it load the saved **raw model weights**; it does not compute or restore EMA weights. If you have a genuine EMA checkpoint, use that instead. For a downloadable model release, package actual files rather than links to local training directories.

### Prepare inference metadata

Inference uses a **JSON object**, not the training JSONL format. Create `metadata.json` following this illustrative example and replace the image path with an existing input:

```json
{
  "sample_0001": {
    "id": "t2/example_case_slice000.png",
    "prompt": "Generate a T1-weighted MRI image from the input T2-weighted MRI image, preserving anatomical correspondence."
  }
}
```

The outer key determines the output filename (`sample_0001.png`). The inner `id` is the source image path relative to `--image_root`. Use unique outer keys without directory separators, and avoid a trailing `.png` in those keys because the script appends it. Ground-truth targets are not required for generation.

### Generate all samples on one GPU

```bash
CUDA_VISIBLE_DEVICES=0 \
torchrun --nnodes=1 --nproc_per_node=1 \
  --master_addr=127.0.0.1 --master_port=28007 \
  eval/gen/gen_images_mp_imgedit.py \
  --output_dir ./outputs/synermed \
  --metadata_file ./metadata.json \
  --image_root "$SYNERMED_DATA_ROOT/slice" \
  --split-count 1 \
  --split-index 0 \
  --max_latent_size 64 \
  --model-path "$INFER_MODEL_DIR"
```

Use an image root appropriate to your own dataset if the inputs are outside SynerMed. Even single-GPU inference must use `torchrun` because the script initializes distributed communication.

The default guidance settings are `cfg_text_scale=4.0` and `cfg_img_scale=1.5`. The script uses 50 denoising steps and `timestep_shift=3.0` internally. Existing output images are skipped; select a new output directory when changing checkpoints or generation settings.

### Split a larger inference job

Your development example uses `--split-count 2 --split-index 1`. That processes only the **second half** of the metadata, in JSON insertion order. To cover the complete dataset, also run index `0`, or use `--split-count 1 --split-index 0` as above.

For multi-GPU inference on one node, expose the desired GPUs and increase `--nproc_per_node`. Each process loads a full model and processes its portion of the selected metadata. When launching separate `torchrun` jobs concurrently, give each job a distinct `--master_port`.

The current implementation forces `think=False` inside the inference loop, so `--think` does not enable reasoning output. It also crops the generated image using `getbbox()` before saving; the resulting PNG may have different dimensions from the input.

## Evaluation and reproducibility

The included `EVAL.md` and most of `eval/` describe inherited BAGEL benchmarks. They do not provide the SynerMed medical PSNR/SSIM/MAE evaluation protocol. Exact reproduction of the medical results additionally needs the released test split, source/target pairing metadata, metric implementation and preprocessing settings, and the corresponding fine-tuned checkpoints.

The package currently provides one mixed-training configuration rather than separate Stage I and Stage II recipes. Do not interpret this example as specifying the paper's inter-stage weight transfer, freezing policy, or stage-specific training duration.

## Citation

If you use SynerMedGen or SynerMed in your research, please cite:

```bibtex
@inproceedings{zhao2026synermedgen,
  title     = {Synergizing Medical Multimodal Understanding with Generation via Task Alignment},
  author    = {Zhao, Weiren and Dong, Yi and Chen, Cheng},
  booktitle = {International Conference on Machine Learning},
  year      = {2026}
}
```

## Acknowledgments and license

This implementation builds on [BAGEL](https://github.com/ByteDance-Seed/Bagel) and retains its Apache-2.0 license and source-file notices. See `LICENSE`. Dataset permissions and source-dataset terms are separate from the code license; refer to the [SynerMed dataset page](https://huggingface.co/datasets/tuy55/SynerMed) for data-specific terms.
