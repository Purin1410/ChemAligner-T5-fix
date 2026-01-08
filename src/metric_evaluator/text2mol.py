
import os
import re
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, Optional, List, Dict, Any

import numpy as np
import torch

from difflib import SequenceMatcher
from transformers import BertTokenizerFast
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

import nltk
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit import DataStructs, RDLogger

import selfies as sf
from selfies.exceptions import DecoderError
from tqdm import tqdm

try:
    from .text2mol import Text2MolMLP
except ImportError:
    Text2MolMLP = None

RDLogger.DisableLog("rdApp.*")


# -----------------------------
# Helpers (top-level, picklable)
# -----------------------------
_SELFIES_TOKEN_RE = re.compile(r"\[[^\[\]]+\]")

def _normalize_selfies(s: Optional[str]) -> str:
    """
    Make model output safer for SELFIES decoding:
    - remove common special tokens like <pad>, </s>, <s>
    - extract only bracket tokens: [C][O]...
    - join tokens (no spaces)
    """
    if not s:
        return ""
    s = s.strip()
    # remove common seq2seq special tokens
    for tok in ("<pad>", "</s>", "<s>", "<unk>"):
        s = s.replace(tok, "")
    # keep only SELFIES bracket tokens
    toks = _SELFIES_TOKEN_RE.findall(s)
    return "".join(toks).strip()


def _decode_selfies_to_smiles(selfies_str: str) -> Optional[str]:
    """Decode SELFIES -> SMILES, return None on failure."""
    try:
        if not selfies_str:
            return None
        return sf.decoder(selfies_str)
    except (DecoderError, Exception):
        return None


def _canonical_smiles_from_smiles(smi: Optional[str]) -> Optional[str]:
    """Canonicalize SMILES (non-isomeric, canonical). Return None if invalid."""
    if smi is None:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)


def _worker_chunk_molecule_metrics(
    gt_selfies_chunk: List[str],
    pred_selfies_chunk: List[str],
    morgan_r: int,
    compute_fcd_pairs: bool,
) -> Tuple[int, int, float, float, float, List[str], List[str]]:
    """
    Compute molecule sums on a chunk.
    We FILTER validity=0 samples by only accumulating on PAIR_VALID:
      - gt_valid: GT decodes + canonicalizes + mol exists
      - pred_valid: Pred decodes + canonicalizes + mol exists
      - pair_valid: gt_valid & pred_valid

    Returns:
      gt_valid_count,
      pair_valid_count,
      sum_maccs, sum_rdk, sum_morgan,
      valid_smiles_gt, valid_smiles_pred (for FCD, only pair_valid)
    """
    gt_valid = 0
    pair_valid = 0
    sum_maccs = 0.0
    sum_rdk = 0.0
    sum_morgan = 0.0
    valid_smiles_gt: List[str] = []
    valid_smiles_pred: List[str] = []

    for gt_sf_raw, pr_sf_raw in zip(gt_selfies_chunk, pred_selfies_chunk):
        gt_sf = _normalize_selfies(gt_sf_raw)
        pr_sf = _normalize_selfies(pr_sf_raw)
        if not gt_sf:
            continue

        gt_smiles = _decode_selfies_to_smiles(gt_sf)
        gt_can = _canonical_smiles_from_smiles(gt_smiles) if gt_smiles is not None else None
        if gt_can is None:
            continue

        m_gt = Chem.MolFromSmiles(gt_can)
        if m_gt is None:
            continue

        gt_valid += 1  # gt hợp lệ -> mới tính validity denominator

        pr_smiles = _decode_selfies_to_smiles(pr_sf)
        pr_can = _canonical_smiles_from_smiles(pr_smiles) if pr_smiles is not None else None
        if pr_can is None:
            # validity(sample)=0 => bị loại khỏi FTS/FCD
            continue

        m_pr = Chem.MolFromSmiles(pr_can)
        if m_pr is None:
            continue

        # pair valid => accumulate similarities
        pair_valid += 1

        maccs_gt = MACCSkeys.GenMACCSKeys(m_gt)
        maccs_pr = MACCSkeys.GenMACCSKeys(m_pr)
        sum_maccs += DataStructs.FingerprintSimilarity(
            maccs_gt, maccs_pr, metric=DataStructs.TanimotoSimilarity
        )

        rdk_gt = Chem.RDKFingerprint(m_gt)
        rdk_pr = Chem.RDKFingerprint(m_pr)
        sum_rdk += DataStructs.FingerprintSimilarity(
            rdk_gt, rdk_pr, metric=DataStructs.TanimotoSimilarity
        )

        fp_gt = AllChem.GetMorganFingerprint(m_gt, morgan_r)
        fp_pr = AllChem.GetMorganFingerprint(m_pr, morgan_r)
        sum_morgan += DataStructs.TanimotoSimilarity(fp_gt, fp_pr)

        if compute_fcd_pairs:
            valid_smiles_gt.append(gt_can)
            valid_smiles_pred.append(pr_can)

    return gt_valid, pair_valid, sum_maccs, sum_rdk, sum_morgan, valid_smiles_gt, valid_smiles_pred


class Text2MolMetrics:
    """Compute text and molecule-level metrics, compatible with Mol2Text_translation."""

    def __init__(
        self,
        device: str = "cuda",
        text_model: str = "allenai/scibert_scivocab_uncased",
        eval_text2mol: bool = False,
        fcd_fn=None,
    ) -> None:
        self.text_tokenizer = BertTokenizerFast.from_pretrained(text_model)
        self.device = torch.device(device)
        self.fcd_fn = fcd_fn

        # avoid crashing if nltk resources not present
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            try:
                nltk.download("wordnet", quiet=True)
            except Exception:
                pass
        try:
            nltk.data.find("corpora/omw-1.4")
        except LookupError:
            try:
                nltk.download("omw-1.4", quiet=True)
            except Exception:
                pass

        self.eval_text2mol = eval_text2mol and (Text2MolMLP is not None)
        if self.eval_text2mol:
            self.text2mol_model = Text2MolMLP(
                ninp=768,
                nhid=600,
                nout=300,
                model_name_or_path=text_model,
                cid2smiles_path=os.path.join(os.path.dirname(__file__), "ckpts", "cid_to_smiles.pkl"),
                cid2vec_path=os.path.join(os.path.dirname(__file__), "ckpts", "test.txt"),
            )
            self.text2mol_model.load_state_dict(
                torch.load(
                    os.path.join(os.path.dirname(__file__), "ckpts", "test_outputfinal_weights.320.pt"),
                    map_location=self.device,
                ),
                strict=False,
            )
            self.text2mol_model.to(self.device)
        else:
            self.text2mol_model = None
            self.eval_text2mol = False

    def __norm_smile_to_isomeric(self, smi: str) -> str:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        return Chem.MolToSmiles(mol, isomericSmiles=True)

    def __canonical_smiles(self, smi: Optional[str]) -> Optional[str]:
        return _canonical_smiles_from_smiles(smi)

    def __selfies_to_smiles(self, selfies_str: str) -> Optional[str]:
        selfies_str = _normalize_selfies(selfies_str)
        return _decode_selfies_to_smiles(selfies_str)

    def __seq_similarity(self, s1: str, s2: str) -> float:
        return SequenceMatcher(None, s1, s2).ratio()

    def __compute_fingerprint_sims_filtered(
        self,
        smiles_gt: List[Optional[str]],
        smiles_pred: List[Optional[str]],
        morgan_r: int = 2,
    ) -> Dict[str, Optional[float]]:
        """
        FILTER validity=0 samples:
          - Skip invalid GT entirely (as before)
          - Only accumulate FTS on pairs where BOTH gt & pred are valid
        validity = pair_valid / gt_valid  (bounded [0,1])
        """
        if len(smiles_gt) != len(smiles_pred):
            raise ValueError("smiles_gt and smiles_pred must have the same length.")

        gt_valid = 0
        pair_valid = 0
        maccs_sims: List[float] = []
        rdk_sims: List[float] = []
        morgan_sims: List[float] = []

        for gt_smi, pr_smi in zip(smiles_gt, smiles_pred):
            cg = self.__canonical_smiles(gt_smi)
            if cg is None:
                continue

            m_gt = Chem.MolFromSmiles(cg)
            if m_gt is None:
                continue
            
            gt_valid += 1

            cp = self.__canonical_smiles(pr_smi)
            if cp is None:
                # validity(sample)=0 -> exclude from FTS
                continue
            m_pr = Chem.MolFromSmiles(cp)
            if m_pr is None:
                continue

            pair_valid += 1

            maccs_gt = MACCSkeys.GenMACCSKeys(m_gt)
            maccs_pr = MACCSkeys.GenMACCSKeys(m_pr)
            maccs_sims.append(DataStructs.FingerprintSimilarity(maccs_gt, maccs_pr))

            rdk_gt = Chem.RDKFingerprint(m_gt)
            rdk_pr = Chem.RDKFingerprint(m_pr)
            rdk_sims.append(DataStructs.FingerprintSimilarity(rdk_gt, rdk_pr))

            fp_gt = AllChem.GetMorganFingerprint(m_gt, morgan_r)
            fp_pr = AllChem.GetMorganFingerprint(m_pr, morgan_r)
            morgan_sims.append(DataStructs.TanimotoSimilarity(fp_gt, fp_pr))

        if gt_valid == 0:
            return {"validity": 0.0, "maccs_fts": None, "rdk_fts": None, "morgan_fts": None}

        validity = float(pair_valid / float(gt_valid))
        if pair_valid == 0:
            return {"validity": validity, "maccs_fts": None, "rdk_fts": None, "morgan_fts": None}

        return {
            "validity": validity,
            "maccs_fts": float(np.mean(maccs_sims)) if maccs_sims else None,
            "rdk_fts": float(np.mean(rdk_sims)) if rdk_sims else None,
            "morgan_fts": float(np.mean(morgan_sims)) if morgan_sims else None,
        }

    def __compute_fcd(self, smiles_gt: List[str], smiles_pred: List[str]) -> Optional[float]:
        if self.fcd_fn is None:
            return None
        return float(self.fcd_fn(smiles_gt, smiles_pred))

    def __call__(
        self,
        predictions: List[str],
        references: List[str],
        smiles: Optional[List[str]] = None,
        selfies_gt: Optional[List[str]] = None,
        selfies_pred: Optional[List[str]] = None,
        smiles_gt: Optional[List[Optional[str]]] = None,
        smiles_pred: Optional[List[Optional[str]]] = None,
        text_trunc_length: int = 512,
    ) -> Dict[str, Any]:
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length.")

        meteor_scores = []
        text2mol_scores = []
        exact_match_scores = []
        levenshtein_scores = []

        refs_tokenized = []
        preds_tokenized = []

        if self.eval_text2mol:
            if smiles is None or len(smiles) != len(predictions):
                raise ValueError("For eval_text2mol=True, 'smiles' must match predictions length.")
            zip_iter = zip(references, predictions, smiles)
        else:
            zip_iter = zip(references, predictions)

        for t in tqdm(zip_iter):
            if self.eval_text2mol:
                gt, out, smi = t
            else:
                gt, out = t
                smi = None

            gt_tokens = self.text_tokenizer.tokenize(
                gt,
                truncation=True,
                max_length=text_trunc_length,
                padding="max_length",
            )
            gt_tokens = [tok for tok in gt_tokens if tok not in ["[PAD]", "[CLS]", "[SEP]"]]

            out_tokens = self.text_tokenizer.tokenize(
                out,
                truncation=True,
                max_length=text_trunc_length,
                padding="max_length",
            )
            out_tokens = [tok for tok in out_tokens if tok not in ["[PAD]", "[CLS]", "[SEP]"]]

            refs_tokenized.append([gt_tokens])
            preds_tokenized.append(out_tokens)

            # meteor may fail if wordnet missing -> fallback 0.0
            try:
                meteor_scores.append(meteor_score([gt_tokens], out_tokens))
            except Exception:
                meteor_scores.append(0.0)

            exact_match_scores.append(1.0 if gt.strip() == out.strip() else 0.0)
            levenshtein_scores.append(self.__seq_similarity(gt, out))

            if self.eval_text2mol and self.text2mol_model is not None and smi is not None:
                norm_smi = self.__norm_smile_to_isomeric(smi)
                t2m_score = self.text2mol_model(norm_smi, out, self.device).detach().cpu().item()
                text2mol_scores.append(t2m_score)

        bleu = corpus_bleu(refs_tokenized, preds_tokenized)
        bleu2 = corpus_bleu(refs_tokenized, preds_tokenized, weights=(0.5, 0.5))
        bleu4 = corpus_bleu(refs_tokenized, preds_tokenized, weights=(0.25, 0.25, 0.25, 0.25))

        meteor_mean = float(np.mean(meteor_scores)) if meteor_scores else 0.0
        exact_match = float(np.mean(exact_match_scores)) if exact_match_scores else 0.0
        levenshtein = float(np.mean(levenshtein_scores)) if levenshtein_scores else 0.0

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
        rouge_scores = [scorer.score(out, gt) for gt, out in zip(references, predictions)]
        rouge_1 = float(np.mean([rs["rouge1"].fmeasure for rs in rouge_scores])) if rouge_scores else 0.0
        rouge_2 = float(np.mean([rs["rouge2"].fmeasure for rs in rouge_scores])) if rouge_scores else 0.0
        rouge_l = float(np.mean([rs["rougeL"].fmeasure for rs in rouge_scores])) if rouge_scores else 0.0

        text2mol = float(np.mean(text2mol_scores)) if text2mol_scores else None

        # --- molecule inputs ---
        if (smiles_gt is None or smiles_pred is None) and (selfies_gt is not None and selfies_pred is not None):
            if len(selfies_gt) != len(selfies_pred):
                raise ValueError("selfies_gt and selfies_pred must have the same length.")
            smiles_gt = [self.__selfies_to_smiles(s) for s in selfies_gt]
            smiles_pred = [self.__selfies_to_smiles(s) for s in selfies_pred]

        # --- molecule metrics (FILTER invalid pairs) ---
        if smiles_gt is None or smiles_pred is None:
            validity = None
            maccs_fts = None
            rdk_fts = None
            morgan_fts = None
            fcd = None
        else:
            fp_res = self.__compute_fingerprint_sims_filtered(smiles_gt, smiles_pred)
            validity = fp_res["validity"]
            maccs_fts = fp_res["maccs_fts"]
            rdk_fts = fp_res["rdk_fts"]
            morgan_fts = fp_res["morgan_fts"]

            # FCD only on valid pairs (same filtering)
            fcd = None
            if self.fcd_fn is not None:
                valid_gt: List[str] = []
                valid_pr: List[str] = []
                for gt_smi, pr_smi in zip(smiles_gt, smiles_pred):
                    cg = self.__canonical_smiles(gt_smi)
                    if cg is None:
                        continue
                    m_gt = Chem.MolFromSmiles(cg)
                    if m_gt is None:
                        continue

                    cp = self.__canonical_smiles(pr_smi)
                    if cp is None:
                        continue
                    m_pr = Chem.MolFromSmiles(cp)
                    if m_pr is None:
                        continue

                    valid_gt.append(cg)
                    valid_pr.append(cp)

                if valid_gt and valid_pr:
                    fcd = self.__compute_fcd(valid_gt, valid_pr)

        return {
            "bleu": bleu,
            "bleu2": bleu2,
            "bleu4": bleu4,
            "rouge1": rouge_1,
            "rouge2": rouge_2,
            "rougeL": rouge_l,
            "meteor": meteor_mean,
            "exact_match": exact_match,
            "levenshtein": levenshtein,
            "text2mol": text2mol,
            "validity": validity,
            "maccs_fts": maccs_fts,
            "rdk_fts": rdk_fts,
            "morgan_fts": morgan_fts,
            "fcd": fcd,
        }

    def compute_molecule_metrics_only(
        self,
        selfies_gt: List[str],
        selfies_pred: List[str],
        morgan_r: int = 2,
        num_proc: int = 0,
        chunk_size: int = 1024,
        compute_fcd: bool = True,
    ) -> Dict[str, Any]:
        """
        Fast molecule-only metrics:
          validity, maccs_fts, rdk_fts, morgan_fts, fcd

        IMPORTANT: Filter validity=0 samples before aggregating FTS/FCD
          - FTS/FCD only computed on PAIR_VALID
          - validity = pair_valid / gt_valid
        """
        if len(selfies_gt) != len(selfies_pred):
            raise ValueError("selfies_gt and selfies_pred must have the same length.")

        n = len(selfies_gt)
        if n == 0:
            return {
                "validity": 0.0,
                "maccs_fts": None,
                "rdk_fts": None,
                "morgan_fts": None,
                "fcd": None,
            }

        if num_proc <= 0:
            num_proc = os.cpu_count() or 1

        compute_fcd_pairs = compute_fcd and (self.fcd_fn is not None)

        chunks: List[Tuple[List[str], List[str]]] = []
        for i in range(0, n, chunk_size):
            chunks.append((selfies_gt[i:i + chunk_size], selfies_pred[i:i + chunk_size]))

        gt_valid_total = 0
        pair_valid_total = 0
        sum_maccs = 0.0
        sum_rdk = 0.0
        sum_morgan = 0.0
        all_gt: List[str] = []
        all_pr: List[str] = []

        if num_proc <= 1 or len(chunks) == 1:
            for gt_chunk, pr_chunk in chunks:
                gv, pv, sm, sr, smg, vgt, vpr = _worker_chunk_molecule_metrics(
                    gt_chunk, pr_chunk, morgan_r, compute_fcd_pairs
                )
                gt_valid_total += gv
                pair_valid_total += pv
                sum_maccs += sm
                sum_rdk += sr
                sum_morgan += smg
                if compute_fcd_pairs:
                    all_gt.extend(vgt)
                    all_pr.extend(vpr)
        else:
            ctx = mp.get_context("spawn")
            max_workers = min(num_proc, len(chunks))
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
                futures = [
                    ex.submit(_worker_chunk_molecule_metrics, gt_chunk, pr_chunk, morgan_r, compute_fcd_pairs)
                    for gt_chunk, pr_chunk in chunks
                ]
                for fu in futures:
                    gv, pv, sm, sr, smg, vgt, vpr = fu.result()
                    gt_valid_total += gv
                    pair_valid_total += pv
                    sum_maccs += sm
                    sum_rdk += sr
                    sum_morgan += smg
                    if compute_fcd_pairs:
                        all_gt.extend(vgt)
                        all_pr.extend(vpr)

        if gt_valid_total == 0:
            return {
                "validity": 0.0,
                "maccs_fts": None,
                "rdk_fts": None,
                "morgan_fts": None,
                "fcd": None,
            }

        validity = float(pair_valid_total / float(gt_valid_total))
        # print("validity: ",validity)

        # If no valid pairs => FTS/FCD are None (because we filtered all validity=0 samples out)
        if pair_valid_total == 0:
            return {
                "validity": validity,
                "maccs_fts": None,
                "rdk_fts": None,
                "morgan_fts": None,
                "fcd": None,
            }

        maccs_fts = float(sum_maccs / float(pair_valid_total))
        rdk_fts = float(sum_rdk / float(pair_valid_total))
        morgan_fts = float(sum_morgan / float(pair_valid_total))

        fcd = None
        if compute_fcd_pairs and all_gt and all_pr:
            fcd = float(self.fcd_fn(all_gt, all_pr))

        return {
            "validity": validity,
            "maccs_fts": maccs_fts,
            "rdk_fts": rdk_fts,
            "morgan_fts": morgan_fts,
            "fcd": fcd,
        }