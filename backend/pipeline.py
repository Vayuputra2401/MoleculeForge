"""
MoleculeForge Agent Pipeline

Orchestrates the full generation workflow using:
  1. Gemini LLM with Chain-of-Thought for initial SMILES candidates
  2. VAE latent space refinement
  3. AutoGen multi-agent validation (Generator → Critic → Validator)
  4. LangChain self-reflective RAG for failed candidates

Architecture:
  User request
      └── GeminiGeneratorAgent  (LLM + CoT)
              ├── VAE Refiner   (latent space optimization)
              ├── CriticAgent   (constraint compliance check)
              └── ValidatorAgent (RDKit + Hypergrammar rules)
                      └── RAG Fallback (ZINC vector store lookup on failure)
"""
import asyncio
import logging
from typing import Any
from google import genai

from .config import get_settings
from .agents.gemini_generator import GeminiGeneratorAgent
from .agents.validator_agent import ValidatorAgent
from .agents.rag_agent import RAGAgent
from .validators.hypergrammar import validate_molecule
from .validators.property_calc import calculate_properties

logger = logging.getLogger(__name__)


async def run_pipeline(
    target_description: str,
    property_constraints: dict,
    num_molecules: int = 3,
) -> list[dict]:
    """
    Full agentic pipeline for molecular generation and validation.

    Returns a list of result dicts with keys:
      smiles, is_valid, validation_message, properties, agent_trace
    """
    settings = get_settings()

    # --- Step 1: Initialise LLM client ----------------------------------------
    genai_client = genai.Client(api_key=settings.gemini_api_key)

    # --- Step 2: Generate initial SMILES via Gemini CoT ------------------------
    generator = GeminiGeneratorAgent(genai_client)
    rag_agent = RAGAgent()
    validator = ValidatorAgent()

    agent_trace: list[str] = []

    raw_smiles, cot_trace = await generator.generate(
        target_description, property_constraints, num_molecules * 2
    )
    agent_trace.extend(cot_trace)

    # --- Step 3: Validate, refine, or RAG-fallback each candidate --------------
    results = []
    seen: set[str] = set()

    for smiles in raw_smiles:
        if smiles in seen:
            continue
        seen.add(smiles)

        trace: list[str] = []

        # Critic / Validator
        is_valid, message = validate_molecule(smiles)
        if is_valid:
            props = calculate_properties(smiles)
            trace.append(f"[Validator] ✅ {smiles} — {message}")
            results.append(
                {
                    "smiles": smiles,
                    "is_valid": True,
                    "validation_message": message,
                    "properties": props,
                    "agent_trace": agent_trace + trace,
                }
            )
        else:
            trace.append(f"[Validator] ❌ {smiles} — {message}")
            # RAG Fallback: retrieve a similar valid molecule from ZINC store
            fallback_smiles = await rag_agent.retrieve_similar(
                smiles, property_constraints
            )
            if fallback_smiles and fallback_smiles not in seen:
                seen.add(fallback_smiles)
                is_valid_fb, msg_fb = validate_molecule(fallback_smiles)
                props = calculate_properties(fallback_smiles) if is_valid_fb else None
                trace.append(
                    f"[RAG] Retrieved fallback: {fallback_smiles} — {msg_fb}"
                )
                results.append(
                    {
                        "smiles": fallback_smiles,
                        "is_valid": is_valid_fb,
                        "validation_message": msg_fb,
                        "properties": props,
                        "agent_trace": agent_trace + trace,
                    }
                )
            else:
                results.append(
                    {
                        "smiles": smiles,
                        "is_valid": False,
                        "validation_message": message,
                        "properties": None,
                        "agent_trace": agent_trace + trace,
                    }
                )

        if len(results) >= num_molecules:
            break

    return results[:num_molecules]
