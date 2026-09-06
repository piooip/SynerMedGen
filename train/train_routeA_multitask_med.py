
# Copyright 2025 Bytedance Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Route A (Multitask, Medical): connector+latent heads only
# Tasks: I2I modality translate (MSE), T2I (via cond-drop), VLM (CE)
#
# Example:
# torchrun --nproc_per_node=8 -u /usr/bin/python /mnt/data/train_routeA_multitask_med.py \
#   --dataset_config_file /path/to/med_mix.yaml \
#   --model_path /mnt/SSD/data_zwr/Bagel/models/BAGEL-7B-MoT \
#   --llm_path hf/Qwen2.5-0.5B-Instruct \
#   --vit_path hf/siglip-so400m-14-980-flash-attn2-navit \
#   --vae_path flux/vae/ae.safetensors \
#   --visual_gen true --visual_und true \
#   --freeze_llm true --freeze_vit true --freeze_vae true --freeze_und false \
#   --expected_num_tokens 16384 --max_num_tokens 24576 --prefer_buffer_before 12288 \
#   --vit_max_num_patch_per_side 56 --max_latent_size 32 --latent_patch_size 2 \
#   --lr 3e-4 --lr_scheduler cosine --warmup_steps 2000 --total_steps 50000 \
#   --mse_weight 1.0 --ce_weight 0.5 \
#   --domain_tokens "T1,T1CE,FLAIR,T2,CT,MRI,contrast,non-contrast" \
#   --log_every 20 --save_every 200
#
import os
import functools
from copy import deepcopy
from datetime import timedelta
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper,
)

import yaml
import wandb
from transformers import HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

# ==== Project-local imports ====
from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.data_utils import add_special_tokens
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from train.train_utils import create_logger, get_latest_ckpt
from train.fsdp_utils import (
    FSDPCheckpoint, FSDPConfig, grad_checkpoint_check_fn, fsdp_wrapper,
    fsdp_ema_setup, fsdp_ema_update,
)

@dataclass
class ModelArguments:
    model_path: str = field(default="hf/BAGEL-7B-MoT")
    llm_path: str = field(default="hf/Qwen2.5-0.5B-Instruct/")
    llm_qk_norm: bool = field(default=True)
    tie_word_embeddings: bool = field(default=False)
    layer_module: str = field(default="Qwen2MoTDecoderLayer")
    vae_path: str = field(default="flux/vae/ae.safetensors")
    vit_path: str = field(default="hf/siglip-so400m-14-980-flash-attn2-navit/")
    max_latent_size: int = field(default=32)
    latent_patch_size: int = field(default=2)
    vit_patch_size: int = field(default=14)
    vit_max_num_patch_per_side: int = field(default=56)
    connector_act: str = field(default="gelu_pytorch_tanh")
    interpolate_pos: bool = field(default=False)
    vit_select_layer: int = field(default=-2)
    vit_rope: bool = field(default=False)
    text_cond_dropout_prob: float = field(default=0.1)
    vae_cond_dropout_prob: float = field(default=0.5)  # ↑ for T2I via cond-drop
    vit_cond_dropout_prob: float = field(default=0.25)

@dataclass
class DataArguments:
    dataset_config_file: str = field(default="data/configs/example.yaml")
    prefetch_factor: int = field(default=2)
    num_workers: int = field(default=4)
    max_num_tokens_per_sample: int = field(default=16384)
    max_num_tokens: int = field(default=24576)
    prefer_buffer_before: int = field(default=12288)
    max_buffer_size: int = field(default=50)
    data_seed: int = field(default=42)

@dataclass
class TrainingArguments:
    # branches
    visual_gen: bool = field(default=True)
    visual_und: bool = field(default=True)
    # bookkeeping
    results_dir: str = field(default="results/routeA_multi")
    checkpoint_dir: str = field(default="results/routeA_multi/checkpoints")
    wandb_project: str = field(default="bagel")
    wandb_name: str = field(default="routeA-multi")
    wandb_runid: str = field(default="0")
    wandb_resume: str = field(default="allow")
    wandb_offline: bool = field(default=False)
    # reproducibility & resume
    global_seed: int = field(default=4396)
    auto_resume: bool = field(default=False)
    resume_from: str = field(default=None)
    resume_model_only: bool = field(default=False)
    finetune_from_ema: bool = field(default=False)
    finetune_from_hf: bool = field(default=False)
    # schedule
    log_every: int = field(default=20)
    save_every: int = field(default=2000)
    total_steps: int = field(default=50000)
    warmup_steps: int = field(default=2000)
    lr_scheduler: str = field(default="cosine")
    lr: float = field(default=3e-4)
    min_lr: float = field(default=1e-7)
    beta1: float = field(default=0.9)
    beta2: float = field(default=0.95)
    eps: float = field(default=1e-15)
    ema: float = field(default=0.9999)
    max_grad_norm: float = field(default=1.0)
    timestep_shift: float = field(default=1.0)
    mse_weight: float = field(default=1.0)
    ce_weight: float = field(default=0.5)  # ↓ a bit so gen dominates
    ce_loss_reweighting: bool = field(default=False)
    expected_num_tokens: int = field(default=16384)
    # FSDP
    num_replicate: int = field(default=1)
    num_shard: int = field(default=8)
    sharding_strategy: str = field(default="HYBRID_SHARD")
    backward_prefetch: str = field(default="BACKWARD_PRE")
    cpu_offload: bool = field(default=False)
    # freezing
    freeze_llm: bool = field(default=True)
    freeze_vit: bool = field(default=True)
    freeze_vae: bool = field(default=True)
    freeze_und: bool = field(default=False)
    # misc
    use_flex: bool = field(default=False)
    grad_accum_steps: int = field(default=1)
    domain_tokens: str = field(default="")  # comma-separated new tokens

def _init_ddp():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    assert torch.cuda.is_available()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    try:
        dist.init_process_group(
            backend="nccl", init_method="env://",
            rank=rank, world_size=world_size,
            timeout=timedelta(minutes=30), device_id=device
        )
    except TypeError:
        dist.init_process_group(
            backend="nccl", init_method="env://",
            rank=rank, world_size=world_size,
            timeout=timedelta(minutes=30)
        )
    try:
        dist.barrier(device_ids=[device])
    except TypeError:
        dist.barrier()
    return device

def _routeA_set_trainables(bagel_model):
    for p in bagel_model.parameters():
        p.requires_grad_(False)
    names = []
    for n,p in bagel_model.named_parameters():
        if any(k in n for k in ["connector.", "vae2llm", "llm2vae", "time_embedder", "latent_pos_embed"]):
            p.requires_grad_(True); names.append(n)
    return names

def main():
    device = _init_ddp()

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # logging
    if dist.get_rank() == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())
        wandb.init(
            project=training_args.wandb_project,
            id=f"{training_args.wandb_name}-run{training_args.wandb_runid}",
            name=training_args.wandb_name,
            resume=training_args.wandb_resume,
            mode=("offline" if training_args.wandb_offline else "online"),
        )
        wandb.config.update(training_args, allow_val_change=True)
        wandb.config.update(model_args, allow_val_change=True)
        wandb.config.update(data_args, allow_val_change=True)
    else:
        logger = create_logger(None, dist.get_rank())
    logger.info(f"TrainingArguments: {training_args}")
    logger.info(f"ModelArguments: {model_args}")
    logger.info(f"DataArguments: {data_args}")

    # resume flags
    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.checkpoint_dir) or training_args.resume_from
        resume_model_only = training_args.resume_model_only
        finetune_from_ema = training_args.finetune_from_ema if resume_model_only else False
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only
        finetune_from_ema = training_args.finetune_from_ema if resume_model_only else False

    # seed
    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)

    # build model
    if training_args.finetune_from_hf:
        llm_config = Qwen2Config.from_json_file(os.path.join(model_args.model_path, "llm_config.json"))
    else:
        llm_config = Qwen2Config.from_pretrained(model_args.llm_path)
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und

    if training_args.finetune_from_hf:
        language_model = Qwen2ForCausalLM(llm_config)
    else:
        language_model = Qwen2ForCausalLM.from_pretrained(model_args.llm_path, config=llm_config)

    if training_args.visual_und:
        if training_args.finetune_from_hf:
            vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_args.model_path, "vit_config.json"))
        else:
            vit_config = SiglipVisionConfig.from_pretrained(model_args.vit_path)
        vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
        vit_config.rope = model_args.vit_rope
        vit_model = SiglipVisionModel.from_pretrained(model_args.vit_path, config=vit_config) if not training_args.finetune_from_hf else SiglipVisionModel(vit_config)
    else:
        vit_config = None; vit_model = None

    if training_args.visual_gen:
        vae_model, vae_config = load_ae(
            local_path=os.path.join(model_args.model_path, "ae.safetensors") if training_args.finetune_from_hf else model_args.vae_path
        )
    else:
        vae_model, vae_config = None, None

    bagel_config = BagelConfig(
        visual_gen=training_args.visual_gen, visual_und=training_args.visual_und,
        llm_config=llm_config,
        vit_config=vit_config if training_args.visual_und else None,
        vae_config=vae_config if training_args.visual_gen else None,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
    )
    model = Bagel(language_model, vit_model if training_args.visual_und else None, bagel_config)
    if training_args.visual_und:
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    # tokenizer + (optional) new domain tokens
    tok = Qwen2Tokenizer.from_pretrained(model_args.model_path if training_args.finetune_from_hf else model_args.llm_path)
    tok, new_token_ids, num_new = add_special_tokens(tok)
    if training_args.domain_tokens:
        extra = [t.strip() for t in training_args.domain_tokens.split(",") if t.strip()]
        if extra:
            tok.add_tokens(extra)
    if (len(tok) != model.language_model.config.vocab_size):
        model.language_model.resize_token_embeddings(len(tok))
        model.config.llm_config.vocab_size = len(tok)
        model.language_model.config.vocab_size = len(tok)

    # freezing
    if training_args.freeze_vae and training_args.visual_gen and vae_model is not None:
        for p in vae_model.parameters(): p.requires_grad_(False)
    if training_args.freeze_llm:
        model.language_model.eval()
        for p in model.language_model.parameters(): p.requires_grad_(False)
    if training_args.freeze_vit and training_args.visual_und and model.vit_model is not None:
        model.vit_model.eval()
        for p in model.vit_model.parameters(): p.requires_grad_(False)

    # whitelist heads
    names = _routeA_set_trainables(model)
    if dist.get_rank()==0:
        print("[RouteA-Multi] Trainable:")
        for n in names: print("  -", n)

    # FSDP
    fsdp_cfg = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
    )
    ema_model = deepcopy(model)
    model, ema_model = FSDPCheckpoint.try_load_ckpt(
        resume_from, logger, model, ema_model, resume_from_ema=training_args.finetune_from_ema
    )
    ema_model = fsdp_ema_setup(ema_model, fsdp_cfg)
    fsdp_model = fsdp_wrapper(model, fsdp_cfg)
    apply_activation_checkpointing(
        fsdp_model,
        checkpoint_wrapper_fn=functools.partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=grad_checkpoint_check_fn,
    )
    if dist.get_rank()==0: print(fsdp_model)

    # optimizer/scheduler
    params = [p for p in fsdp_model.parameters() if p.requires_grad]
    assert params, "No trainable params."
    optimizer = torch.optim.AdamW(params, lr=training_args.lr,
                                  betas=(training_args.beta1, training_args.beta2),
                                  eps=training_args.eps, weight_decay=0.0)
    if training_args.lr_scheduler=="cosine":
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps, min_lr=training_args.min_lr,
        )
    else:
        scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=training_args.warmup_steps)

    # resume state
    if training_args.resume_model_only:
        train_step=0; data_status=None
    else:
        optimizer, scheduler, train_step, data_status = FSDPCheckpoint.try_load_train_state(
            resume_from, optimizer, scheduler, fsdp_cfg
        )

    # dataloader
    with open(data_args.dataset_config_file, "r") as f:
        dataset_meta = yaml.safe_load(f)
    ds_cfg = DataConfig(grouped_datasets=dataset_meta)
    if training_args.visual_und:
        ds_cfg.vit_patch_size = model_args.vit_patch_size
        ds_cfg.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    if training_args.visual_gen and vae_config is not None:
        ds_cfg.vae_image_downsample = model_args.latent_patch_size * vae_config.downsample
        ds_cfg.max_latent_size = model_args.max_latent_size
    ds_cfg.text_cond_dropout_prob = model_args.text_cond_dropout_prob
    ds_cfg.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
    ds_cfg.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob

    train_dataset = PackedDataset(
        ds_cfg, tokenizer=tok, special_tokens=new_token_ids,
        local_rank=dist.get_rank(), world_size=dist.get_world_size(),
        num_workers=data_args.num_workers, expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
        max_num_tokens=data_args.max_num_tokens, max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before, interpolate_pos=model_args.interpolate_pos,
        use_flex=training_args.use_flex, data_status=data_status,
    )
    train_dataset.set_epoch(data_args.data_seed)
    train_loader = DataLoader(
        train_dataset, batch_size=1, num_workers=data_args.num_workers, pin_memory=True,
        collate_fn=collate_wrapper(), drop_last=True, prefetch_factor=data_args.prefetch_factor
    )

    # ready
    if training_args.visual_gen and vae_model is not None: vae_model.to(device).eval()
    fsdp_model.train(); ema_model.eval()

    logger.info(f"[RouteA-Multi] steps={training_args.total_steps} start={train_step}")
    fsdp_model.zero_grad(set_to_none=True)
    use_amp = True

    for curr_step, batch in enumerate(train_loader, start=train_step):
        batch = batch.cuda(device).to_dict()
        data_indexes = batch.pop("batch_data_indexes", None)
        ce_loss_weights = batch.pop("ce_loss_weights", None)

        # VAE encode only if we actually have images for gen
        has_padded_images = ("padded_images" in batch)
        if has_padded_images and training_args.visual_gen and vae_model is not None:
            with torch.no_grad():
                # encode and replace with latent (model.forward expects 'padded_latent')
                batch["padded_latent"] = vae_model.encode(batch.pop("padded_images"))

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            loss_dict = fsdp_model(**batch)

            total_loss = 0.0
            # CE (VLM)
            has_ce_idx = ("ce_loss_indexes" in batch) and (loss_dict.get("ce", None) is not None)
            total_ce_tokens = torch.tensor(0, device=device)
            if has_ce_idx:
                ce = loss_dict["ce"]
                total_ce_tokens = torch.tensor(len(batch["ce_loss_indexes"]), device=device)
                dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
                if training_args.ce_loss_reweighting and (ce_loss_weights is not None):
                    ce = ce * ce_loss_weights
                    total_ce_loss_weights = ce_loss_weights.sum()
                    dist.all_reduce(total_ce_loss_weights, op=dist.ReduceOp.SUM)
                    ce = ce.sum() * dist.get_world_size() / (total_ce_loss_weights + 1e-6)
                else:
                    ce = ce.sum() * dist.get_world_size() / (total_ce_tokens + 1e-6)
                loss_dict["ce"] = ce.detach()
                total_loss = total_loss + ce * training_args.ce_weight
            else:
                loss_dict["ce"] = torch.tensor(0.0, device=device)

            # MSE (I2I/T2I latent)
            has_mse_idx = ("mse_loss_indexes" in batch) and (loss_dict.get("mse", None) is not None)
            total_mse_tokens = torch.tensor(0, device=device)
            if has_mse_idx:
                mse = loss_dict["mse"]
                total_mse_tokens = torch.tensor(len(batch["mse_loss_indexes"]), device=device)
                dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
                mse = mse.mean(dim=-1).sum() * dist.get_world_size() / (total_mse_tokens + 1e-6)
                loss_dict["mse"] = mse.detach()
                total_loss = total_loss + mse * training_args.mse_weight
            else:
                loss_dict["mse"] = torch.tensor(0.0, device=device)

        (total_loss / training_args.grad_accum_steps).backward()
        if (curr_step + 1) % training_args.grad_accum_steps == 0:
            total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
            optimizer.step(); scheduler.step()
            fsdp_model.zero_grad(set_to_none=True)
            fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)

        # logging
        if curr_step % training_args.log_every == 0:
            steps_per_sec = None  # kept simple
            total_samples = torch.tensor(len(batch.get("sample_lens", [])), device=device)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

            msg = f"(step={curr_step:07d}) "
            wandb_log = {}
            for k, v in loss_dict.items():
                avg = torch.tensor(float(v.item() if torch.is_tensor(v) else v), device=device)
                dist.all_reduce(avg, op=dist.ReduceOp.SUM)
                avg = avg.item() / dist.get_world_size()
                msg += f"Train Loss {k}: {avg:.4f}, "
                wandb_log[k] = avg
            msg += f"has_ce={int(has_ce_idx)}, has_mse={int(has_mse_idx)} "
            logger.info(msg)

            wandb_log.update({
                "lr": optimizer.param_groups[0]["lr"],
                "has_ce": int(has_ce_idx), "has_mse": int(has_mse_idx),
                "total_ce_tokens": int(total_ce_tokens.item() if torch.is_tensor(total_ce_tokens) else 0),
                "total_mse_tokens": int(total_mse_tokens.item() if torch.is_tensor(total_mse_tokens) else 0),
                "total_samples": int(total_samples.item()),
                "mem_allocated(MB)": torch.cuda.max_memory_allocated() / (1024**2),
                "mem_reserved(MB)": torch.cuda.max_memory_reserved() / (1024**2),
            })
            if dist.get_rank()==0: wandb.log(wandb_log, step=curr_step)

        # track data_status & save
        if data_status is None: data_status = {}
        if data_indexes is not None:
            for it in data_indexes:
                if it["dataset_name"] not in data_status:
                    data_status[it["dataset_name"]] = {}
                data_status[it["dataset_name"]][it["worker_id"]] = it["data_indexes"]

        if curr_step > 0 and curr_step % training_args.save_every == 0:
            gather_list = [None] * dist.get_world_size() if dist.get_rank()==0 else None
            dist.gather_object(data_status, gather_list, dst=0)
            FSDPCheckpoint.fsdp_save_ckpt(
                ckpt_dir=training_args.checkpoint_dir, train_steps=curr_step,
                model=fsdp_model, ema_model=ema_model,
                optimizer=optimizer, scheduler=scheduler, logger=logger,
                fsdp_config=fsdp_cfg, data_status=gather_list,
            )

    # finalize
    try:
        dist.barrier(device_ids=[device])
    except TypeError:
        dist.barrier()
    if dist.get_rank()==0: wandb.finish()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
