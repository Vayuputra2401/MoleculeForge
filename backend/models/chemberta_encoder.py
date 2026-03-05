"""
Pretrained Molecular Encoder (ChemBERTa from Hugging Face)
==========================================================
Uses DeepChem/ChemBERTa-77M-MLM — a RoBERTa-based transformer pre-trained
on 77 million SMILES strings — to produce dense molecular embeddings.

Why ChemBERTa over Morgan fingerprints?
  - Contextualised representations: ChemBERTa encodes chemical context, not
    just atom-pair presence, producing richer latent vectors.
  - Pretrained knowledge: trained on 77M molecules, it generalises to
    unseen scaffolds better than heuristic fingerprints.
  - Direct SMILES input: no graph conversion required; drop-in for any
    SMILES string.

Model card: https://huggingface.co/DeepChem/ChemBERTa-77M-MLM

Embedding dimension : 384 (hidden_size of the checkpoint)
Output              : mean-pooled last hidden state → numpy float32 array
"""
import logging
import numpy as np
from typing import Optional

log = logging.getLogger(__name__)

# Cached model + tokenizer (loaded once per process)
_tokenizer = None
_model = None
_device = None


def _load_model():
    """Lazy-load ChemBERTa from Hugging Face Hub on first call."""
    global _tokenizer, _model, _device

    if _model is not None:
        return

    try:
        import torch
        from transformers import AutoTokenizer, AutoModel

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("[ChemBERTa] Loading DeepChem/ChemBERTa-77M-MLM on %s …", _device)

        _tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        _model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        _model.eval()
        _model.to(_device)

        log.info("[ChemBERTa] Model loaded (hidden_size=%d)", _model.config.hidden_size)
    except Exception as e:
        log.warning("[ChemBERTa] Failed to load model: %s — falling back to Morgan fingerprints", e)
        _model = "failed"


def embed_smiles(smiles: str) -> Optional[np.ndarray]:
    """
    Encode a single SMILES string into a dense embedding vector.

    Returns:
        numpy float32 array of shape (hidden_size,) or None on failure.
    """
    _load_model()
    if _model == "failed" or _model is None:
        return _morgan_fallback(smiles)

    try:
        import torch

        inputs = _tokenizer(
            smiles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(_device)

        with torch.no_grad():
            outputs = _model(**inputs)

        # Mean-pool over the sequence dimension (excluding padding)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        token_embeddings = outputs.last_hidden_state
        mean_embedding = (token_embeddings * attention_mask).sum(1) / attention_mask.sum(1)
        return mean_embedding.squeeze(0).cpu().numpy().astype(np.float32)
    except Exception as e:
        log.warning("[ChemBERTa] embed_smiles failed for '%s': %s", smiles, e)
        return _morgan_fallback(smiles)


def embed_batch(smiles_list: list[str]) -> np.ndarray:
    """
    Batch-encode a list of SMILES strings.

    Returns:
        numpy float32 array of shape (N, hidden_size)
    """
    _load_model()
    if _model == "failed" or _model is None:
        return np.array([_morgan_fallback(s) for s in smiles_list], dtype=np.float32)

    try:
        import torch

        inputs = _tokenizer(
            smiles_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(_device)

        with torch.no_grad():
            outputs = _model(**inputs)

        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        token_embeddings = outputs.last_hidden_state
        embeddings = (token_embeddings * attention_mask).sum(1) / attention_mask.sum(1)
        return embeddings.cpu().numpy().astype(np.float32)
    except Exception as e:
        log.warning("[ChemBERTa] embed_batch failed: %s", e)
        return np.array([_morgan_fallback(s) for s in smiles_list], dtype=np.float32)


def _morgan_fallback(smiles: str) -> np.ndarray:
    """2048-bit Morgan fingerprint as fallback when ChemBERTa is unavailable."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            return np.array(list(fp), dtype=np.float32)
    except Exception:
        pass
    return np.zeros(2048, dtype=np.float32)
