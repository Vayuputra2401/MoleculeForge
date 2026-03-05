# Agent Architecture Documentation

## Overview

The MoleculeForge agentic pipeline orchestrates four specialised AI components to generate, validate, and refine drug-like molecular candidates.

```
User Request
    │
    ▼
┌───────────────────┐
│  GeminiGenerator  │  ← Fine-tuned Gemini API + Chain-of-Thought Prompting
│  Agent            │    Reasons about scaffold/pharmacophore, then emits SMILES
└───────┬───────────┘
        │  raw SMILES candidates (×N)
        ▼
┌───────────────────┐
│  ValidatorAgent   │  ← Two-stage multi-agent validation
│   ├─ Critic       │    • Heuristic property compliance check
│   └─ Validator    │    • Deterministic RDKit + hypergrammar rules
└───────┬───────────┘
        │ PASS ──────────────────────────────────┐
        │ FAIL                                   │
        ▼                                        │
┌───────────────────┐                            │
│  RAGAgent         │  ← Self-reflective RAG      │
│  (ZINC + FAISS)   │    • Morgan fingerprint     │
│                   │      similarity search      │
│                   │    • Property-filtered       │
└───────┬───────────┘      nearest neighbour      │
        │ fallback SMILES                         │
        ▼                                         │
┌──────────────────────────────────────────────┐  │
│  PropertyCalculator  (RDKit descriptors)      │◄─┘
│  Returns: QED, LogP, MW, TPSA, SA Score, …  │
└──────────────────────────────────────────────┘
        │
        ▼
  JSON response to React frontend
```

---

## Agent Roles

### 1. GeminiGeneratorAgent (`agents/gemini_generator.py`)
| Property | Value |
|---|---|
| Model | `gemini-2.0-flash` (production: fine-tuned variant) |
| Strategy | Chain-of-Thought (CoT) |
| Prompt | Two-phase: `## Reasoning` → `## Molecules` (JSON) |
| Output | List of SMILES + CoT trace lines |

**CoT sections captured:**
- Pharmacophore analysis
- Scaffold selection rationale
- Constraint satisfaction reasoning

---

### 2. ValidatorAgent (`agents/validator_agent.py`)
Two sequential sub-agents:

**Structural Validator** (deterministic)
- Lipinski Rule of Five check
- Problematic substructure scan (PAINS-like SMARTS)
- Peroxide / charged atom detection

**Critic Agent** (heuristic)
- Logs calculated QED, LogP, MW against user constraints
- Flags molecules near boundary values

---

### 3. RAGAgent (`agents/rag_agent.py`)
Self-reflective RAG loop triggered on validation failure:

1. **ZINC index built** from Hugging Face dataset (`graphs-datasets/ZINC`, 50k rows via `scripts/download_zinc.py`)
2. **ChemBERTa embedding** (`DeepChem/ChemBERTa-77M-MLM`) computed for the failed molecule — 384-dim contextualised vector
3. FAISS flat L2 search retrieves top-K nearest neighbours (index cached at `data/zinc_faiss.index`)
4. Property filter applied to select first passing candidate
5. Selected molecule re-enters the ValidatorAgent pipeline

**Encoder:** [DeepChem/ChemBERTa-77M-MLM on Hugging Face](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM)
**Dataset:** [graphs-datasets/ZINC on Hugging Face](https://huggingface.co/datasets/graphs-datasets/ZINC)
**Fallback:** If ChemBERTa unavailable, Morgan fingerprints (2048-bit, radius 2) are used automatically

---

### 4. PropertyCalculator (`validators/property_calc.py`)
Final stage — always applied to accepted molecules:
- Molecular formula, MW, LogP, QED, TPSA, LabuteASA
- Lipinski violations count, Veber (rotatable bonds)
- Synthetic Accessibility (SA) score via RDKit
- Estimated LogS (Yalkowsky equation)

---

## Inter-Agent Message Flow

```python
# Simplified pseudocode showing the message passing
smiles_list, cot_trace = await gemini_agent.generate(description, constraints)

for smiles in smiles_list:
    is_valid, msg    = hypergrammar.validate_molecule(smiles)    # ValidatorAgent
    if is_valid:
        props = calculate_properties(smiles)
        yield Result(smiles, valid=True, props=props, trace=cot_trace)
    else:
        fallback = await rag_agent.retrieve_similar(smiles, constraints)  # RAGAgent
        yield Result(fallback or smiles, ...)
```

---

## Extension Points

| Upgrade | How |
|---|---|
| Full AutoGen wiring | Replace `ValidatorAgent` with `autogen.AssistantAgent` pair |
| Larger ZINC index | Swap FAISS flat → IVF index; load full ZINC-250K |
| Fine-tuned Gemini | Set `model="tunedModels/<your-model-id>"` in `gemini_generator.py` |
| LangGraph orchestration | Wrap agents as LangGraph nodes for stateful, cyclical flows |
| Observability | Add LangSmith `callbacks` to `GeminiGeneratorAgent` constructor |
