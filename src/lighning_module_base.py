import math
from typing import Any, List

import lightning as pl
import torch
from torch import optim

from src.backbones.lang.chemaligner_t5 import T5ForConditionalGeneration
from src.metric_evaluator.text2mol import Text2MolMetrics

# Optional FCD dependency
from fcd import get_fcd

def fcd_fn(smiles_gt: List[str], smiles_pred: List[str]) -> float:
    return float(get_fcd(smiles_gt, smiles_pred))


class T5Model(pl.LightningModule):
    """
    Baseline seq2seq LightningModule:
    - No contrastive loss
    - No config parsing here
    - All hyperparams must be injected into args in train.py
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        # Tokenizer object must be assigned in train.py: model.tokenizer = tokenizer
        self.tokenizer = None

        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            args.t5.pretrained_model_name_or_path
        )

        # Metrics evaluator
        effective_fcd = fcd_fn if (bool(args.eval_compute_fcd) and fcd_fn is not None) else None
        self.metric_evaluator = Text2MolMetrics(
            eval_text2mol=bool(args.run_text2mol_metrics),
            fcd_fn=effective_fcd,
        )

        # Final buffers (strings) for epoch metrics
        self._val_pred_selfies: List[str] = []
        self._val_gt_selfies: List[str] = []

        # GPU mega-batch buffers (tensors)
        self._buf_input_ids: List[torch.Tensor] = []
        self._buf_attention_mask: List[torch.Tensor] = []
        self._buf_gt_selfies: List[List[str]] = []
        self._buf_size: int = 0

    def resize_token_embeddings(self, vocab_size: int) -> None:
        self.t5_model.resize_token_embeddings(vocab_size)

    def _prepare_inputs(self, batch):
        return batch["input_ids"], batch["attention_mask"], batch["labels"]

    def forward(self, input_ids, attention_mask, labels=None):
        # Replace padding token id with -100 so it is ignored by LM loss
        labels = labels.clone()
        labels[labels == int(self.args.pad_token_id)] = -100

        out = self.t5_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return out.loss, out.logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = self._prepare_inputs(batch)
        loss, _ = self(input_ids, attention_mask, labels)

        self.log("train/lm_loss", loss, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_start(self):
        self._val_pred_selfies = []
        self._val_gt_selfies = []
        self._buf_input_ids, self._buf_attention_mask, self._buf_gt_selfies = [], [], []
        self._buf_size = 0

    def _append_buf(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gt_selfies: List[str],
    ) -> None:
        bs = int(input_ids.size(0))
        self._buf_input_ids.append(input_ids)
        self._buf_attention_mask.append(attention_mask)
        self._buf_gt_selfies.append(gt_selfies)
        self._buf_size += bs

    def _flush_buf_generate(self) -> None:
        if self._buf_size == 0:
            return
        if self.tokenizer is None:
            raise ValueError("model.tokenizer is None. Please set model.tokenizer = tokenizer in train.py")

        mega_input_ids = torch.cat(self._buf_input_ids, dim=0)
        mega_attention_mask = torch.cat(self._buf_attention_mask, dim=0)

        mega_gt: List[str] = []
        for part in self._buf_gt_selfies:
            mega_gt.extend(part)

        # Apply max_samples cap (0 means unlimited)
        if int(self.args.eval_max_samples) > 0:
            remaining = int(self.args.eval_max_samples) - len(self._val_gt_selfies)
            if remaining <= 0:
                self._buf_input_ids, self._buf_attention_mask, self._buf_gt_selfies = [], [], []
                self._buf_size = 0
                return
            mega_input_ids = mega_input_ids[:remaining]
            mega_attention_mask = mega_attention_mask[:remaining]
            mega_gt = mega_gt[:remaining]

        use_amp = bool(self.args.eval_use_amp) and mega_input_ids.is_cuda

        with torch.inference_mode():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred_ids = self.t5_model.generate(
                        input_ids=mega_input_ids,
                        attention_mask=mega_attention_mask,
                        max_length=int(self.args.eval_max_length),
                        num_beams=int(self.args.eval_num_beams),
                    )
            else:
                pred_ids = self.t5_model.generate(
                    input_ids=mega_input_ids,
                    attention_mask=mega_attention_mask,
                    max_length=int(self.args.eval_max_length),
                    num_beams=int(self.args.eval_num_beams),
                )

        pred_selfies = self.tokenizer.batch_decode(pred_ids)
        pred_selfies = [
            s.replace("<unk>", "")
             .replace("<pad>", "")
             .replace("</s>", "")
             .replace("<bom>", "")
             .replace("<eom>", "")
             .strip()
            for s in pred_selfies
        ]
        gt_selfies = [(s or "").strip() for s in mega_gt]

        for p, g in zip(pred_selfies, gt_selfies):
            if p and g:
                self._val_pred_selfies.append(p)
                self._val_gt_selfies.append(g)

        self._buf_input_ids, self._buf_attention_mask, self._buf_gt_selfies = [], [], []
        self._buf_size = 0

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = self._prepare_inputs(batch)
        loss, _ = self(input_ids, attention_mask, labels)

        self.log("val/lm_loss", loss, prog_bar=True, logger=True)
        self.log("eval_loss", loss, prog_bar=False, logger=False)

        # Only compute molecule metrics for lang2mol batches
        if "selfies" not in batch:
            return

        gt_selfies = batch["selfies"]
        if isinstance(gt_selfies, str):
            gt_selfies = [gt_selfies]
        gt_selfies = [(s or "").strip() for s in gt_selfies]

        self._append_buf(input_ids, attention_mask, gt_selfies)

        target_bs = int(self.args.eval_batch_size)
        if self._buf_size >= target_bs:
            self._flush_buf_generate()

    def _ddp_all_gather_object(self, obj: Any) -> List[Any]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            gathered = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(gathered, obj)
            return gathered
        return [obj]

    def on_validation_epoch_end(self):
        # Flush remaining examples
        self._flush_buf_generate()

        gathered_preds = self._ddp_all_gather_object(self._val_pred_selfies)
        gathered_gts = self._ddp_all_gather_object(self._val_gt_selfies)

        if not self.trainer.is_global_zero:
            return

        all_preds: List[str] = []
        all_gts: List[str] = []
        for part in gathered_preds:
            all_preds.extend(part)
        for part in gathered_gts:
            all_gts.extend(part)

        if not all_preds or not all_gts:
            return

        mol_metrics = self.metric_evaluator.compute_molecule_metrics_only(
            selfies_gt=all_gts,
            selfies_pred=all_preds,
            morgan_r=2,
            num_proc=int(self.args.eval_num_proc),
            chunk_size=int(self.args.eval_chunk_size),
            compute_fcd=bool(self.args.eval_compute_fcd),
        )

        for k, v in mol_metrics.items():
            if v is None:
                continue
            self.log(f"val_metric/{k}", v, prog_bar=False, logger=True, sync_dist=False)

        self.log("val_metric/n_samples", float(len(all_preds)), prog_bar=False, logger=True, sync_dist=False)

    def configure_optimizers(self):
        # Total optimization steps
        total_steps = int(self.args.max_epochs) * int(self.args.train_data_len)
        warmup_steps = int(float(self.args.warmup_ratio) * total_steps)

        # Optimizer selection from args (injected from config)
        opt_name = str(getattr(self.args, "optimizer_name", "adamw")).lower()

        if opt_name == "sgd":
            optimizer = optim.SGD(
                self.parameters(),
                lr=float(self.args.lr),
                momentum=float(getattr(self.args, "optimizer_momentum", 0.9)),
                weight_decay=float(getattr(self.args, "optimizer_weight_decay", 0.0)),
            )
        else:
            optimizer = optim.AdamW(
                self.parameters(),
                lr=float(self.args.lr),
                weight_decay=float(getattr(self.args, "optimizer_weight_decay", 0.01)),
            )

        scheduler = {
            "scheduler": self._cosine_scheduler(optimizer, total_steps, warmup_steps),
            "name": "learning_rate",
            "interval": "step",
        }
        return [optimizer], [scheduler]

    @staticmethod
    def _cosine_scheduler(optimizer, training_steps: int, warmup_steps: int):
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / max(1, warmup_steps)
            progress = float(current_step - warmup_steps) / max(1, training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def generate_captioning(
        self,
        inputs,
        max_length: int = 512,
        num_beams: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        decoder_start_token_id: int = 0,
        eos_token_id: int = 1,
        pad_token_id: int = 0,
    ) -> List[str]:
        """
        Utility generation function.
        Expects `self.tokenizer` to be set externally.
        """
        if self.tokenizer is None:
            raise ValueError("model.tokenizer is None. Please set model.tokenizer = tokenizer in train.py")

        input_ids, attention_mask, _ = self._prepare_inputs(inputs)
        out_ids = self.t5_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_start_token_id=decoder_start_token_id,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        decoded = self.tokenizer.batch_decode(out_ids)
        decoded = [
            s.replace("<unk>", "")
             .replace("<pad>", "")
             .replace("</s>", "")
             .replace("<bom>", "")
             .replace("<eom>", "")
             .strip()
            for s in decoded
        ]
        return decoded
