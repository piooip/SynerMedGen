#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-file VLM MCQ evaluator for your JSONL format.
- Loads JSONL lines with {"image":[...], "conversations":[...]}.
- First image is the reference (not an option). Images #2–#5 are A/B/C/D.
- Immediately computes accuracy (overall and per-pair if meta exists) after eval.
- Saves results.jsonl and results.txt under --out-dir.

Usage example:
torchrun --nproc_per_node=8 eval_vlm_jsonl.py \
  --jsonl-path /path/to/your.jsonl \
  --image-root  /path/to/image_root \
  --out-dir     /path/to/out_dir \
  --model-path  hf/BAGEL-7B-MoT \
  --max-new-tokens 100
"""

import argparse
import itertools
import json
import os
import random
import re
from collections import defaultdict, Counter
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# Your existing helpers
from eval.vlm.utils import load_model_and_tokenizer, build_transform, process_conversation


# -----------------------------
# Collate & Distributed Sampler
# -----------------------------
def collate_fn(batches):
    questions = [_['question'] for _ in batches]
    images = [_['images'] for _ in batches]
    conversations = [_['conversations'] for _ in batches]
    answers = [_['answer'] for _ in batches]          # 'A'/'B'/'C'/'D'
    data_ids = [_['data_id'] for _ in batches]        # id or line index
    meta = [_['meta'] for _ in batches]               # optional fields
    return questions, images, conversations, answers, data_ids, meta


class InferenceSampler(torch.utils.data.Sampler):
    def __init__(self, size: int):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]
        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


# -----------------------------
# Utilities
# -----------------------------
IMG_TOKEN = "<image>"
LETTER = {"A", "B", "C", "D"}
LETTER_RE = re.compile(r"\b([ABCD])\b", re.I)

def ensure_image_tokens(prompt: str, n_images: int) -> str:
    """Ensure the first <image> placeholder expands to exactly n_images tokens."""
    if prompt.count(IMG_TOKEN) == n_images:
        return prompt
    if IMG_TOKEN not in prompt:
        return (IMG_TOKEN * n_images) + "\n" + prompt
    return prompt.replace(IMG_TOKEN, IMG_TOKEN * n_images, 1)

def extract_letter(text: str) -> str:
    """Robustly extract A|B|C|D from model output."""
    if not isinstance(text, str):
        text = str(text)
    t = text.strip().upper()
    if len(t) == 1 and t in LETTER:
        return t
    m = LETTER_RE.findall(t)
    if m:
        return m[-1].upper()
    for ch in LETTER:
        if f"({ch})" in t or f"OPTION {ch}" in t or f"ANSWER {ch}" in t:
            return ch
    return t  # fallback: return raw text for debugging


# -----------------------------
# Dataset for your JSONL
# -----------------------------
class VLMJSONLDataset(torch.utils.data.Dataset):
    """
    JSONL line example:
    {
      "image": ["t1/...png", "tgt/...png", "tgt/...png", "tgt/...png", "tgt/...png"],
      "conversations": [
        {"from":"human", "value":"<image> ... [A,B,C,D]"},
        {"from":"gpt",   "value":"D"}
      ],
      // optional: "id", "pair": {"src":"t1","tgt":"t1ce"}, "case_id", "slice_idx"
    }
    """
    def __init__(self, image_root: str, jsonl_path: str):
        self.root = Path(image_root)
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                rels = obj["image"]
                convs = obj["conversations"]
                assert isinstance(rels, list) and len(rels) >= 5, "image list must be [ref, A, B, C, D]"
                assert isinstance(convs, list) and len(convs) >= 2, "need at least human & gpt turns"

                # human prompt & ground-truth
                human_msg = next((c["value"] for c in convs if c.get("from") == "human"), None)
                assert human_msg is not None, "missing human message"
                human_msg = ensure_image_tokens(human_msg, len(rels))

                gt_msg = next((c["value"] for c in convs if c.get("from") == "gpt"), "")
                gt_letter = extract_letter(gt_msg)

                meta = {}
                for k in ["id", "pair", "case_id", "slice_idx"]:
                    if k in obj:
                        meta[k] = obj[k]

                self.items.append({
                    "rel_paths": rels,
                    "question": human_msg,
                    "answer": gt_letter,
                    "meta": meta,
                    "line_id": obj.get("id", line_id),
                })

        if not self.items:
            raise RuntimeError(f"No samples loaded from {jsonl_path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        # Load all images
        pil_images = []
        for rel in it["rel_paths"]:
            p = (self.root / rel).resolve()
            img = Image.open(p).convert("RGB")
            pil_images.append(img)

        images, conversation = process_conversation(pil_images, it["question"])
        return {
            "question": it["question"],
            "images": images,
            "conversations": conversation,
            "answer": it["answer"],
            "data_id": it["line_id"],
            "meta": it["meta"],
        }


# -----------------------------
# Evaluation (inference + metrics)
# -----------------------------
def evaluate(args, model, tokenizer, new_token_ids, image_transform):
    random.seed(args.seed)

    dataset = VLMJSONLDataset(args.image_root, args.jsonl_path)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    outputs = []
    for _, (questions, images, conversations, answers, data_ids, metas) in tqdm(
        enumerate(dataloader), total=len(dataloader)
    ):
        # batch size = 1
        pred_text = model.chat(
            tokenizer,
            new_token_ids,
            image_transform,
            images=images[0],
            prompt=conversations[0],
            max_length=args.max_new_tokens,
        )
        pred_letter = extract_letter(pred_text)

        outputs.append({
            "question": questions[0],
            "pred": pred_letter,
            "gt": answers[0],
            "data_id": data_ids[0],
            "meta": metas[0],
            "raw": pred_text,
        })

    # sync & gather
    torch.distributed.barrier()
    world_size = torch.distributed.get_world_size()
    merged = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged, json.dumps(outputs))
    merged = [json.loads(_) for _ in merged]
    merged = list(itertools.chain.from_iterable(merged))

    if torch.distributed.get_rank() != 0:
        return

    # Save raw predictions
    os.makedirs(args.out_dir, exist_ok=True)
    path_jsonl = os.path.join(args.out_dir, "results.jsonl")
    with open(path_jsonl, "w", encoding="utf-8") as w:
        for item in merged:
            w.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[Eval] Results saved to {path_jsonl}")

    # ---- Metrics: Overall, per-pair (optional), confusion matrix ----
    def norm_letter(x: str) -> str:
        t = extract_letter(x)
        return t if t in LETTER else t  # keep raw if not a clean letter (for debugging)

    total, correct = 0, 0
    per_pair = defaultdict(lambda: [0, 0])   # [correct, total]
    conf = Counter()                         # (gt,pred) counts

    for x in merged:
        gt = norm_letter(x["gt"])
        pd = norm_letter(x["pred"])

        total += 1
        if pd in LETTER and gt in LETTER and pd == gt:
            correct += 1

        gshow = gt if gt in LETTER else "OTHER"
        pshow = pd if pd in LETTER else "OTHER"
        conf[(gshow, pshow)] += 1

        pair = x.get("meta", {}).get("pair")
        if pair and isinstance(pair, dict):
            key = f"{pair.get('src')}->{pair.get('tgt')}"
            per_pair[key][1] += 1
            if pd in LETTER and gt in LETTER and pd == gt:
                per_pair[key][0] += 1

    acc = correct / total if total else 0.0
    print(f"[Overall] ACC = {acc:.6f} ({correct}/{total})")

    # Write summary
    path_txt = os.path.join(args.out_dir, "results.txt")
    with open(path_txt, "w", encoding="utf-8") as w:
        w.write(f"[Overall] ACC = {acc:.6f} ({correct}/{total})\n")
        if per_pair:
            w.write("\n[Per-Pair ACC]\n")
            for k in sorted(per_pair.keys()):
                c, t = per_pair[k]
                w.write(f"{k}: { (c/t if t else 0.0):.6f} ({c}/{t})\n")

        # Confusion matrix (A/B/C/D/OTHER)
        labels = ["A","B","C","D","OTHER"]
        w.write("\n[Confusion Matrix counts GT x Pred]\n")
        w.write("\t" + "\t".join(labels) + "\n")
        for g in labels:
            row = [g] + [str(conf.get((g,p), 0)) for p in labels]
            w.write("\t".join(row) + "\n")

    print(f"[Eval] Summary saved to {path_txt}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-path", type=str, required=True, help="Path to your VLM JSONL file")
    parser.add_argument("--image-root", type=str, required=True, help="Root dir that contains modality folders (t1/, t1ce/, etc.)")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", type=str, default="hf/BAGEL-7B-MoT/")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # init distributed
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=int(os.getenv("WORLD_SIZE", "1")),
        rank=int(os.getenv("RANK", "0")),
    )
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", 0)))

    # load model
    model, tokenizer, new_token_ids = load_model_and_tokenizer(args)
    image_transform = build_transform()
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[test] total_params: {total_params:.3f}B")

    evaluate(args, model, tokenizer, new_token_ids, image_transform)
