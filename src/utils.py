"""Copyright 2023 by @jet981217. All rights reserved."""
from typing import List
from argparse import Namespace
import torch

import sys
import subprocess
from pathlib import Path
from typing import Any

import yaml



def infer_eval_config_path_from_train(train_config_path: str) -> str:
    """
    Infer evaluation config path from training config path by
    replacing the suffix '_train' with '_eval' in the filename.

    Example:
      src/configs/config_lpm24_train.yaml
      -> src/configs/config_lpm24_eval.yaml
    """
    train_path = Path(train_config_path)

    name = train_path.name
    if "_train" in name:
        eval_name = name.replace("_train", "_eval")
    else:
        # Fallback: keep the original name if no '_train' suffix is found
        # This avoids breaking when users provide non-standard config names
        eval_name = name

    return str(train_path.with_name(eval_name))

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
    s = str(x).strip()
    return [s] if s else []


def collect_ckpt_paths(output_folder: str) -> List[str]:
    """
    Collect all .ckpt files in output_folder and return a sorted list of paths.
    """
    ckpt_dir = Path(str(output_folder))
    ckpt_paths = sorted([str(p) for p in ckpt_dir.glob("*.ckpt")])
    return ckpt_paths


def build_resolved_eval_yaml(eval_yaml_path: str, ckpt_paths: List[str], out_dir: str) -> str:
    """
    Load eval YAML, replace eval_init.checkpoint_paths, then dump to a new YAML file.
    Return path to the resolved YAML.
    """
    eval_yaml_path = str(eval_yaml_path)
    out_dir = str(out_dir)

    with open(eval_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("eval_init", {})
    cfg["eval_init"]["checkpoint_paths"] = normalize_checkpoint_list(ckpt_paths)

    # Save resolved YAML inside the training output folder for traceability
    out_path = Path(out_dir) / f"resolved_{Path(eval_yaml_path).name}"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    return str(out_path)


def run_eval_subprocess(resolved_eval_yaml: str, num_devices: int) -> None:
    """
    Run eval_lang2mol via subprocess.
    - Single GPU: python eval_lang2mol.py --model_config <yaml>
    - Multi GPU: torchrun --nproc_per_node=<num_devices> eval_lang2mol.py --model_config <yaml>
    """
    num_devices = int(num_devices or 1)

    if num_devices <= 1:
        cmd = [sys.executable, "eval_lang2mol.py", "--model_config", resolved_eval_yaml]
    else:
        cmd = ["torchrun", f"--nproc_per_node={num_devices}", "eval_lang2mol.py", "--model_config", resolved_eval_yaml]

    print("[Eval] Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)





def enable_tf32_if_tensor_cores_available(device: int | None = None, verbose: bool = True) -> bool:
    """
    Check GPU availability and (if Tensor Cores are present) enable TF32 /
    set_float32_matmul_precision("high").

    Returns:
        True if TF32 is enabled, False otherwise
        (no CUDA / no Tensor Cores / fallback case).

    Notes:
    - Call this function once at the beginning of the program before training/evaluation.
    - TF32 / Tensor Cores trade numerical precision for performance.
      Bit-for-bit reproducibility is not guaranteed when TF32 is enabled.
    """
    if not torch.cuda.is_available():
        if verbose:
            print("CUDA is not available — TF32 not enabled.")
        return False

    # Select device (default: current CUDA device)
    if device is None:
        device = torch.cuda.current_device()
    try:
        prop = torch.cuda.get_device_properties(device)
    except Exception as e:
        if verbose:
            print(f"Failed to retrieve device properties: {e}")
        return False

    name = prop.name
    major = prop.major
    minor = prop.minor
    compute_capability = major + minor / 10.0

    # Heuristic: Tensor Cores are available on GPUs with compute capability >= 7.0
    # (Volta, Turing?, Ampere, Ada, Hopper, ...)
    has_tensor_cores = major >= 7

    if verbose:
        print(f"Device {device}: {name}, compute capability {major}.{minor} ({compute_capability})")
        print(f"Tensor Cores detected (heuristic): {has_tensor_cores}")

    if not has_tensor_cores:
        if verbose:
            print("No Tensor Cores detected by heuristic — keeping default settings.")
        return False

    # Enable TF32 / set precision (if supported)
    # If PyTorch supports torch.set_float32_matmul_precision (>= ~2.0), prefer using it.
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
            if verbose:
                print('Called torch.set_float32_matmul_precision("high").')
        except Exception as e:
            if verbose:
                print("Failed to call set_float32_matmul_precision:", e)

    # Backend flags (for older / compatible PyTorch versions)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        # Some PyTorch versions do not expose torch.backends.cuda.matmul
        print("Cannot activate torch.backends.cuda.matmul.allow_tf32")
    try:
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        print("Cannot activate torch.backends.cudnn.allow_tf32")

    if verbose:
        print("TF32 / allow_tf32 has been enabled (if supported by backend).")
        print("WARNING: TF32 trades numerical precision for performance; bit-for-bit reproducibility is lost.")

    return True


def set_nested_attr(obj, key, value):
    if isinstance(value, dict):
        if not hasattr(obj, key):
            setattr(obj, key, Namespace())
        
        for subkey in value:
            set_nested_attr(getattr(obj, key), subkey, value[subkey])
    else:
        setattr(obj, key, value)
class AGG:
    def __init__(
        self,
        device: str,
        alpha: float,
        memory_len: int,
        vocab_size: int,
        token_ids_to_use: List[int],
    ) -> None:
        """Adaptive gradient gating class

        Args:
            device (str):
                Device to handle tensors.
            alpha (float):
                Hyper-parameter to decide rare tokens.
            memory_len (int):
                Steps to log.
            vocab_size (int):
                Total size of vocab for tokenizer.
            token_ids_to_use (List[int]):
                Token ids which contain real words.
                [MASK], [CLS], etc. cannot be part of this list.
        """
        dev = torch.device(device)
        self.__device = dev
        self.__alpha = alpha
        self.__step = 0
        self.__vocab_size = vocab_size

        # Chuẩn hóa token_ids_to_use
        token_ids = torch.as_tensor(token_ids_to_use, dtype=torch.long)
        assert token_ids.min() >= 0
        assert token_ids.max() < vocab_size
        self.__token_ids_to_use = token_ids.to(dev)

        # mask cho các token được dùng
        self.__valid_mask = torch.zeros(vocab_size, dtype=torch.bool, device=dev)
        self.__valid_mask[self.__token_ids_to_use] = True

        # memory_cell: [memory_len, vocab_size]
        self.__memory_cell = torch.zeros(memory_len, vocab_size, device=dev)
        # appearance_rate: [vocab_size]
        self.__appearance_rate = torch.ones(vocab_size, device=dev)

        # các vector gating & mask
        self.__g1_gate_vector = torch.ones(vocab_size, device=dev)
        self.__g2_gate_vector = torch.ones(vocab_size, device=dev)
        self.__rare_mask = torch.zeros(vocab_size, dtype=torch.bool, device=dev)
        self.__very_rare_mask = torch.zeros(vocab_size, dtype=torch.bool, device=dev)

    # -----------------------
    #  core update
    # -----------------------
    def dynamic_rare_token_grouping(self) -> None:
        """Vectorized rare token grouping (không vòng for trên vocab)"""
        dev = self.__appearance_rate.device
        mem = self.__memory_cell

        # Số step thực tế có dữ liệu
        boundary = min(self.__step, mem.size(0))
        if boundary == 0:
            # chưa có dữ liệu để tính mean -> giữ nguyên gate hiện tại
            return

        # mean theo time: [vocab_size]
        memory_mean = mem[:boundary].mean(dim=0)

        # chỉ chuẩn hóa trên token_ids_to_use
        valid_mean = memory_mean[self.__valid_mask]  # [num_valid]
        sum_valid = valid_mean.sum()

        if sum_valid > 0:
            # cập nhật appearance_rate trên các token hợp lệ
            self.__appearance_rate[self.__valid_mask] = valid_mean / sum_valid
        # nếu sum_valid == 0 -> không update, tránh NaN

        # --- Rare tokens ---
        # rare nếu appearance_rate < alpha *và* thuộc token_ids_to_use
        rare_mask = (self.__appearance_rate < self.__alpha) & self.__valid_mask
        self.__rare_mask = rare_mask

        rare_rates = self.__appearance_rate[rare_mask]
        if rare_rates.numel() > 0:
            mean_apperance_rare = rare_rates.mean()
        else:
            # không có rare -> chọn 1.0 để chia an toàn (không ảnh hưởng nhiều)
            mean_apperance_rare = self.__appearance_rate.new_tensor(1.0)

        # G1 gate: default 1, rare token dùng appearance_rate
        self.__g1_gate_vector = torch.ones_like(self.__appearance_rate, device=dev)
        self.__g1_gate_vector[rare_mask] = self.__appearance_rate[rare_mask]

        # --- Very rare tokens ---
        very_rare_mask = (self.__appearance_rate / mean_apperance_rare) < 1
        self.__very_rare_mask = very_rare_mask

        self.__g2_gate_vector = torch.ones_like(self.__appearance_rate, device=dev)
        self.__g2_gate_vector[very_rare_mask] = (
            self.__appearance_rate[very_rare_mask] / mean_apperance_rare
        )

    # -----------------------
    #  logging step
    # -----------------------
    def step_agg(self, input_tokens_batch: torch.Tensor) -> None:
        """Log 1 step: đếm tần suất token trong batch (dùng bincount)."""
        dev = input_tokens_batch.device

        # đảm bảo các buffer ở đúng device (DDP-friendly)
        if self.__memory_cell.device != dev:
            self.__memory_cell = self.__memory_cell.to(dev)
            self.__appearance_rate = self.__appearance_rate.to(dev)
            self.__g1_gate_vector = self.__g1_gate_vector.to(dev)
            self.__g2_gate_vector = self.__g2_gate_vector.to(dev)
            self.__rare_mask = self.__rare_mask.to(dev)
            self.__very_rare_mask = self.__very_rare_mask.to(dev)
            self.__valid_mask = self.__valid_mask.to(dev)
            self.__token_ids_to_use = self.__token_ids_to_use.to(dev)

        # flatten input và lọc các token nằm trong [0, vocab_size)
        flat = input_tokens_batch.reshape(-1)
        flat = flat[(flat >= 0) & (flat < self.__vocab_size)]

        # dùng bincount -> O(#tokens) thay vì O(vocab_size * #tokens)
        counts = torch.bincount(flat, minlength=self.__vocab_size).to(
            device=dev, dtype=self.__memory_cell.dtype
        )

        # ghi vào ô hiện tại trong memory_cell
        row_idx = self.__step % self.__memory_cell.size(0)
        self.__memory_cell[row_idx] = counts
        self.__step += 1

        # cập nhật rare token grouping
        self.dynamic_rare_token_grouping()

    # -----------------------
    #  gating mask
    # -----------------------
    def get_gate_mask(self, target_tokens: List[int] | torch.Tensor) -> torch.Tensor:
        """Get gating mask cho batch các target_tokens.

        Thay vì stack nhiều vector (g1/g2) rồi mean,
        ta dùng: avg = r * g2 + (1 - r) * g1,
        trong đó r = tỉ lệ target_tokens thuộc rare_mask.
        """
        dev = self.__appearance_rate.device

        if not torch.is_tensor(target_tokens):
            target_tokens = torch.as_tensor(target_tokens, device=dev, dtype=torch.long)
        else:
            target_tokens = target_tokens.to(dev)

        # lọc token hợp lệ (tránh -100, pad, v.v. nếu ngoài range)
        mask_valid = (target_tokens >= 0) & (target_tokens < self.__vocab_size)
        target_tokens = target_tokens[mask_valid]

        if target_tokens.numel() == 0:
            # nếu không có token hợp lệ thì trả về g1 (gate mặc định)
            return self.__g1_gate_vector

        # check token nào trong rare_mask
        is_rare_each = self.__rare_mask[target_tokens]          # [N]
        ratio_rare = is_rare_each.float().mean()                # scalar ∈ [0,1]

        # avg_gate = (N_rare/N)*g2 + (1 - N_rare/N)*g1
        gate = ratio_rare * self.__g2_gate_vector + (1.0 - ratio_rare) * self.__g1_gate_vector
        return gate