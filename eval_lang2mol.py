# eval_lang2mol.py
# Single entrypoint for:
#   1) Lang2Mol inference (caption -> SELFIES) and save CSV
#   2) Compute chemical metrics from the merged CSV
#   3) Support evaluating one or many checkpoints from config (YAML list)
#
# Run:
#   python eval_lang2mol.py --model_config src/configs/config_lpm24_train.yaml
#   torchrun --nproc_per_node=<your devices> eval_lang2mol.py --model_config src/configs/config_lpm24_train.yaml
#
# Important:
#   - All runtime knobs should come from config (no CLI overrides except model_config).
#   - Metrics are computed on rank 0 only in DDP.
#   - For multiple checkpoints, results are appended into one summary file.

import os
import csv
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer
from sconf import Config

from src.dataset_module import get_dataloaders
# from src.metric_evaluator.translation_metrics import Mol2Text_translation

# Optional extra metrics
from src.metric_evaluator.text2mol import Text2MolMetrics
from fcd import get_fcd

def fcd_fn(smiles_gt: List[str], smiles_pred: List[str]) -> float:
    return float(get_fcd(smiles_gt, smiles_pred))
from src.lighning_module_base import T5Model  # type: ignore


DATASET_MAP = {
    "lpm-24": "duongttr/LPM-24-extend",
    "lpm-24-extra": "Neeze/LPM-24-extra-extend",
    "lpm-24-smoke": "Neeze/LPM-24-smoke-extend",
    "lpm-24-eval": "Neeze/LPM-24-eval-extend",
    "chebi-20": "duongttr/chebi-20-new",
}


# -----------------------------
# Helpers
# -----------------------------
def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    cur = cfg
    for part in key.split("."):
        if cur is None or not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur


def ensure_ns(root: Namespace, name: str) -> Namespace:
    if not hasattr(root, name) or getattr(root, name) is None:
        setattr(root, name, Namespace())
    return getattr(root, name)


def normalize_checkpoint_list(x: Any) -> List[str]:
    """
    Accept:
      - string path
      - list of string paths
      - None
    Return list[str].
    """
    if x is None:
        return []
    if isinstance(x, str):
        x = x.strip()
        return [x] if x else []
    if isinstance(x, (list, tuple)):
        out = []
        for it in x:
            if it is None:
                continue
            s = str(it).strip()
            if s:
                out.append(s)
        return out
    return [str(x).strip()]


def parse_precision_to_autocast(precision_value: Any) -> str:
    """
    Map config precision to "fp32", "fp16", "bf16".
    Supports common Lightning values:
      - "32"
      - "16-mixed"
      - "bf16-mixed"
      - 32, 16
    """
    if precision_value is None:
        return "fp32"
    s = str(precision_value).lower().strip()
    if "bf16" in s:
        return "bf16"
    if "16" in s:
        return "fp16"
    return "fp32"


def safe_name_from_path(path: str) -> str:
    """
    Turn a checkpoint path into a stable file-friendly stem.
    """
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    stem = stem.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return stem


# -----------------------------
# DDP utils
# -----------------------------
def dist_info() -> Tuple[bool, int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return True, rank, local_rank, world_size
    return False, 0, 0, 1


def dist_init(local_rank: int) -> None:
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)


def is_main(rank: int) -> bool:
    return rank == 0


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def try_rebuild_dataloader_for_ddp(
    dl: DataLoader,
    rank: int,
    world_size: int,
    num_workers: int,
) -> Tuple[DataLoader, bool]:
    dataset = getattr(dl, "dataset", None)
    if dataset is None:
        return dl, False

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    batch_size = getattr(dl, "batch_size", 1)
    collate_fn = getattr(dl, "collate_fn", None)

    kwargs: Dict[str, Any] = dict(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,
        collate_fn=collate_fn,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4

    return DataLoader(**kwargs), True


def merge_csv_parts(final_csv: str, part_files: List[str]) -> None:
    os.makedirs(os.path.dirname(final_csv) or ".", exist_ok=True)
    with open(final_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=["caption", "gt_selfie", "pred_selfie"])
        writer.writeheader()
        for pf in part_files:
            if not os.path.exists(pf):
                continue
            with open(pf, "r", newline="", encoding="utf-8") as fin:
                reader = csv.DictReader(fin)
                for row in reader:
                    writer.writerow(row)


# -----------------------------
# Config -> args (your requested pattern)
# -----------------------------
def fill_args_from_config(args: Namespace, config: Config) -> None:
    """
    Fill args from config so you only edit YAML.
    """
    # Setup evaluate & model
    args.t5 = Namespace()
    args.t5.pretrained_model_name_or_path = config.t5.pretrained_model_name_or_path
    args.cuda = bool(cfg_get(config, "eval_init.cuda", True))
    args.deterministic = bool(cfg_get(config, "eval_init.deterministic", True))
    
    # Dataset
    args.dataset_name = str(cfg_get(config, "dataset_init.dataset_name", "lpm-24"))
    if args.dataset_name not in DATASET_MAP:
        raise ValueError(f"Invalid dataset_name: {args.dataset_name}. Valid: {', '.join(DATASET_MAP.keys())}")
    args.dataset_name_or_path = DATASET_MAP[args.dataset_name]
    args.task = str(cfg_get(config, "dataset_init.task", "lang2mol"))
    args.num_workers = int(cfg_get(config, "dataset_init.num_workers", 8)/config.eval_init.num_devices)
    args.split = str(cfg_get(config, "dataset_init.split", "validation"))
    args.eval_batch_size = int(cfg_get(config, "dataset_init.eval_batch_size", 1))
    
    #  Eval init setup
    args.eval_max_length = int(cfg_get(config, "eval_init.max_length", 512))
    args.num_beams = int(cfg_get(config, "eval_init.num_beams", 1))
    args.use_amp = bool(cfg_get(config, "eval_init.use_amp", False))
    args.max_samples = int(cfg_get(config, "eval_init.max_samples", 0))
    args.chunk_size = int(cfg_get(config, "eval_init.chunk_size", 0))
    args.eval_precision = cfg_get(config, "eval_init.precision", "32")
    args.print_examples = int(cfg_get(config, "eval_init.print_examples", 0))
    args.offline = bool(cfg_get(config, "eval_init.offline", False))
    args.eval_compute_fcd = True
    args.pin_memory = bool(config.dataset_init.pin_memory)
    args.persistent_workers = bool(config.dataset_init.persistent_workers)
    
    args.eval_output_dir = str(cfg_get(config, "eval_init.output_dir", "results/lang2mol_eval"))
    args.summary_file = str(cfg_get(config, "eval_init.summary_file", "metrics_summary.csv"))

    # Checkpoints list
    args.checkpoint_paths = normalize_checkpoint_list(cfg_get(config, "eval_init.checkpoint_paths", None))



# -----------------------------
# Metrics (computed once per checkpoint on rank 0)
# -----------------------------
def evaluate_from_csv(csv_path: str) -> Dict[str, Any]:
    gt_selfies: List[str] = []
    pred_selfies: List[str] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_selfies.append(row["gt_selfie"])
            pred_selfies.append(row["pred_selfie"])

    results: Dict[str, Any] = {}

    try:
        if fcd_fn is not None:
            t2m = Text2MolMetrics(eval_text2mol=True, fcd_fn=fcd_fn)
        res2 = t2m(predictions=pred_selfies,
                    references=gt_selfies,
                    selfies_gt=gt_selfies,
                    selfies_pred=pred_selfies,)
        for k, v in res2.items():
            results[f"text2mol_{k}"] = v
    except Exception as e:
        results["text2mol_error"] = str(e)

    return results


def append_summary_row(summary_csv: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(summary_csv) or ".", exist_ok=True)
    file_exists = os.path.exists(summary_csv)

    # Keep a stable column order: checkpoint first, then metrics keys sorted
    keys = list(row.keys())
    if "checkpoint" in keys:
        keys.remove("checkpoint")
        keys = ["checkpoint"] + sorted(keys)

    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)

        if not file_exists:
            writer.writeheader()

        # Ensure all fields exist for this header
        out = {k: row.get(k, "") for k in keys}
        writer.writerow(out)


# -----------------------------
# Inference for one checkpoint
# -----------------------------
def run_one_checkpoint(
    ckpt_path: str,
    args: Namespace,
    config: Config,
    tokenizer: Any,
    model: Any,
    val_dl: DataLoader,
    device: torch.device,
    is_dist: bool,
    rank: int,
    world_size: int,
    use_true_shard: bool,
) -> Optional[Dict[str, Any]]:
    """
    Returns metrics dict on rank 0, else returns None.
    """

    # Load checkpoint weights
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    ckpt_name = safe_name_from_path(ckpt_path)

    # Output files per checkpoint
    output_dir = os.path.join(args.eval_output_dir, ckpt_name)
    os.makedirs(output_dir, exist_ok=True)
    merged_csv = os.path.join(output_dir, f"{ckpt_name}.pred.csv")

    # Per rank part file
    out_part = merged_csv if not is_dist else f"{os.path.splitext(merged_csv)[0]}.rank{rank}.csv"

    # Autocast
    autocast_enabled = args.cuda and args.eval_precision in ["fp16", "bf16"]
    autocast_dtype = torch.bfloat16 if args.eval_precision == "bf16" else torch.float16

    pbar = tqdm(val_dl, disable=(is_dist and not is_main(rank)))
    with open(out_part, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["caption", "gt_selfie", "pred_selfie"])
        writer.writeheader()

        with torch.inference_mode():
            for step_idx, batch in enumerate(pbar):
                # Fallback sharding if sampler rebuild failed
                if args.cuda and is_dist and (not use_true_shard):
                    if (step_idx % world_size) != rank:
                        continue

                batch = move_to_device(batch, device)
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
                    pred_selfies = model.generate_captioning(batch)
                # print(pred_selfies)


                gt_selfies = batch["selfies"]
                captions = batch.get("caption", [""] * len(gt_selfies))

                # Optional prints (slow)
                if args.print_examples > 0 and (step_idx < args.print_examples) and (is_main(rank) or (not is_dist)):
                    for p, g in zip(pred_selfies[:3], gt_selfies[:3]):
                        print(f"Predict: {p}")
                        print(f"GT: {g}")
                        print("-" * 50)

                writer.writerows(
                    [
                        {"caption": c, "gt_selfie": g, "pred_selfie": p} 
                        for c, g, p in zip(captions, gt_selfies, pred_selfies)
                    ]
                )

    # Merge + metrics only on rank 0
    if args.cuda and is_dist:
        # dist.barrier()

        if is_main(rank):
            part_files = [f"{os.path.splitext(merged_csv)[0]}.rank{r}.csv" for r in range(world_size)]
            merge_csv_parts(merged_csv, part_files)

            metrics = evaluate_from_csv(merged_csv)
            metrics_row = {"checkpoint": ckpt_path}
            metrics_row.update(metrics)

            summary_csv = os.path.join(output_dir, args.summary_file)
            append_summary_row(summary_csv, metrics_row)

            return metrics_row

        # dist.barrier()
        return None

    # Single process
    metrics = evaluate_from_csv(merged_csv)
    metrics_row = {"checkpoint": ckpt_path}
    metrics_row.update(metrics)
    summary_csv = os.path.join(args.eval_output_dir, args.summary_file)
    append_summary_row(summary_csv, metrics_row)
    return metrics_row


# -----------------------------
# Main
# -----------------------------
def main(args: Namespace, config: Config) -> None:
    # Fill everything from config
    fill_args_from_config(args, config)

    if len(args.checkpoint_paths) == 0:
        raise ValueError(
            "eval_init.checkpoint_paths is empty. Please set it in YAML as a string or a list."
        )

    # Basic env
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    # Seed (optional)
    seed = cfg_get(config, "eval_init.seed_everything", None)
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    # DDP init once
    is_dist, rank, local_rank, world_size = dist_info()
    if args.cuda and is_dist:
        dist_init(local_rank)

    # Perf knobs
    if args.cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = not args.deterministic
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    device = torch.device(f"cuda:{local_rank}" if args.cuda else "cpu")

    # Tokenizer
    local_only = args.offline or (is_dist and not is_main(rank))
    tokenizer = AutoTokenizer.from_pretrained(
        args.t5.pretrained_model_name_or_path,
        local_files_only=local_only,
    )

    # Dataloader (built once, reused across checkpoints)
    base_dl = get_dataloaders(
        args,
        tokenizer,
        batch_size=args.eval_batch_size,  # per process (per GPU in DDP)
        num_workers=args.num_workers,
        split=args.split,
        task=args.task,
    )

    use_true_shard = False
    val_dl = base_dl
    if args.cuda and is_dist:
        val_dl, use_true_shard = try_rebuild_dataloader_for_ddp(
            base_dl, rank, world_size, args.num_workers
        )


    args.tokenizer = Namespace()
    args.tokenizer.pad_token_id = tokenizer.pad_token_id

    model = T5Model(args)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    # model.eval()
    model.tokenizer = tokenizer

    # Prepare summary file path and clear it (rank 0 only)
    summary_csv = os.path.join(args.eval_output_dir, args.summary_file)
    if (not is_dist) or is_main(rank):
        os.makedirs(args.eval_output_dir, exist_ok=True)
        if os.path.exists(summary_csv):
            os.remove(summary_csv)

    # Evaluate each checkpoint
    for ckpt_path in args.checkpoint_paths:
        # if args.cuda and is_dist:
        #     dist.barrier()

        if (not os.path.exists(ckpt_path)) and ((not is_dist) or is_main(rank)):
            print(f"[WARN] Checkpoint does not exist: {ckpt_path}")

        metrics_row = run_one_checkpoint(
            ckpt_path=ckpt_path,
            args=args,
            config=config,
            tokenizer=tokenizer,
            model=model,
            val_dl=val_dl,
            device=device,
            is_dist=is_dist,
            rank=rank,
            world_size=world_size,
            use_true_shard=use_true_shard,
        )

        if (not is_dist) or is_main(rank):
            if metrics_row is not None:
                print("=== Evaluation Results ===")
                for k, v in metrics_row.items():
                    print(f"{k}: {v}")
                print("\n")

        if args.cuda and is_dist:
            dist.barrier()

    # Cleanup
    if args.cuda:
        # dist.barrier()
        dist.destroy_process_group()
    print("End")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_config", type=str, default="src/configs/config_lpm24_train.yaml")
    args = parser.parse_args()

    config = Config(args.model_config)
    main(args, config)
