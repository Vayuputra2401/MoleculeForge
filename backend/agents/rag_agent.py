"""
RAGAgent (Self-Reflective RAG via LangChain)
============================================
Provides fallback candidates when the validator rejects a molecule.

Architecture:
  1. ZINC dataset (graphs-datasets/ZINC from Hugging Face) is downloaded and
     embedded using ChemBERTa (DeepChem/ChemBERTa-77M-MLM) — a pretrained
     transformer molecular encoder — into a FAISS vector store.
  2. When a molecule fails validation, RAGAgent retrieves the k nearest
     neighbours from the ZINC store, filters by property constraints, and
     returns the best alternative.
  3. This "self-reflective" loop grounds the pipeline in known-valid ZINC
     molecules whenever the LLM-generated candidate fails.

Hugging Face resources:
  Encoder : DeepChem/ChemBERTa-77M-MLM   (384-dim contextualised embeddings)
  Dataset : graphs-datasets/ZINC          (drug-like molecules, train split)

Offline mode:
  If data/zinc_50k.parquet exists (pre-built by scripts/download_zinc.py),
  that file is used instead of downloading at runtime, keeping cold-start
  fast. The FAISS index is cached in data/zinc_faiss.index.

Future upgrade:
  Swap the flat FAISS store for a hosted Pinecone / Chroma instance and
  index the full ZINC-250K dataset for production-quality retrieval.
"""
import asyncio
import logging
import numpy as np
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).resolve().parents[2] / "data"
PARQUET    = DATA_DIR / "zinc_50k.parquet"
FAISS_IDX  = DATA_DIR / "zinc_faiss.index"
EMBED_DIM  = 384  # ChemBERTa hidden_size; Morgan fallback uses 2048

# Runtime cache
_store: Optional[object] = None   # faiss.Index
_zinc_smiles: list[str] = []
_built: bool = False


def _build_store() -> None:
    """Build FAISS index from ZINC parquet (or download on-the-fly)."""
    global _store, _zinc_smiles, _built

    if _built:
        return
    _built = True

    try:
        import faiss
        from ..models.chemberta_encoder import embed_batch

        # ── Load SMILES ────────────────────────────────────────────────────────
        if PARQUET.exists():
            import pandas as pd
            log.info("[RAGAgent] Loading ZINC from %s", PARQUET)
            df = pd.read_parquet(PARQUET)
            smiles_col = "smiles" if "smiles" in df.columns else df.columns[0]
            smiles_list = df[smiles_col].dropna().tolist()
        else:
            # Fallback: stream from HF at runtime (first 10k for fast cold-start)
            from datasets import load_dataset
            log.info("[RAGAgent] Streaming graphs-datasets/ZINC from Hugging Face …")
            ds = load_dataset("graphs-datasets/ZINC", split="train[:10000]")
            smiles_col = next(
                (c for c in ("smiles", "smi", "SMILES") if c in ds.column_names),
                ds.column_names[0],
            )
            smiles_list = [r[smiles_col] for r in ds if r[smiles_col]]

        # ── Validate with RDKit ────────────────────────────────────────────────
        from rdkit import Chem
        valid = [s for s in smiles_list if Chem.MolFromSmiles(s) is not None]
        log.info("[RAGAgent] %d / %d ZINC SMILES are valid", len(valid), len(smiles_list))
        _zinc_smiles = valid

        # ── Load or build FAISS index ──────────────────────────────────────────
        if FAISS_IDX.exists():
            log.info("[RAGAgent] Loading cached FAISS index from %s", FAISS_IDX)
            _store = faiss.read_index(str(FAISS_IDX))
        else:
            log.info("[RAGAgent] Embedding %d molecules with ChemBERTa …", len(valid))
            # Embed in batches of 128 to avoid OOM
            batch_size = 128
            all_embs = []
            for i in range(0, len(valid), batch_size):
                batch = valid[i : i + batch_size]
                embs = embed_batch(batch)
                all_embs.append(embs)
            embeddings = np.vstack(all_embs).astype(np.float32)

            dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings)
            _store = index

            DATA_DIR.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(FAISS_IDX))
            log.info("[RAGAgent] FAISS index saved (%d molecules, dim=%d)", len(valid), dim)

    except Exception as e:
        log.warning("[RAGAgent] Could not build ZINC store: %s", e)
        _store = None


class RAGAgent:
    """
    Retrieves the nearest ZINC molecule that satisfies property constraints
    when a generated candidate fails validation.
    """

    async def retrieve_similar(
        self, smiles: str, property_constraints: dict, k: int = 5
    ) -> Optional[str]:
        """Async wrapper — runs the blocking FAISS search in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._sync_retrieve, smiles, property_constraints, k
        )

    def _sync_retrieve(
        self, smiles: str, property_constraints: dict, k: int
    ) -> Optional[str]:
        _build_store()

        if _store is None or not _zinc_smiles:
            return None

        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, QED as RDQED
            from ..models.chemberta_encoder import embed_smiles

            query_emb = embed_smiles(smiles)
            if query_emb is None:
                return None

            query = query_emb.reshape(1, -1).astype(np.float32)
            _, indices = _store.search(query, min(k * 10, len(_zinc_smiles)))

            lc  = property_constraints["logP"]
            mwc = property_constraints["molecularWeight"]
            qedc = property_constraints["QED"]

            for idx in indices[0]:
                if idx < 0 or idx >= len(_zinc_smiles):
                    continue
                candidate = _zinc_smiles[idx]
                mol = Chem.MolFromSmiles(candidate)
                if mol is None:
                    continue
                logp = Descriptors.MolLogP(mol)
                mw   = Descriptors.MolWt(mol)
                qed  = RDQED.qed(mol)
                if (
                    lc["min"]  <= logp <= lc["max"]
                    and mwc["min"] <= mw   <= mwc["max"]
                    and qedc["min"] <= qed  <= qedc["max"]
                ):
                    log.info("[RAGAgent] ZINC fallback found: %s", candidate)
                    return candidate

        except Exception as e:
            log.warning("[RAGAgent] Retrieval failed: %s", e)

        return None
