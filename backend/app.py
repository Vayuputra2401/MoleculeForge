"""
MoleculeForge FastAPI Backend
Handles molecular generation, validation, and agent orchestration.
"""
import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from .pipeline import run_pipeline
from .config import get_settings

app = FastAPI(
    title="MoleculeForge API",
    description="AI-powered molecular generation pipeline",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = get_settings()


class GenerationRequest(BaseModel):
    target_description: str
    logp_min: float = 1.0
    logp_max: float = 4.0
    mw_min: float = 200.0
    mw_max: float = 500.0
    qed_min: float = 0.5
    qed_max: float = 0.9
    num_molecules: int = 3


class MoleculeResult(BaseModel):
    smiles: str
    is_valid: bool
    validation_message: str
    properties: Optional[dict] = None
    agent_trace: Optional[list] = None


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/generate", response_model=list[MoleculeResult])
async def generate_molecules(req: GenerationRequest):
    """
    Run the full agentic molecular generation pipeline:
      1. Gemini LLM generation with Chain-of-Thought
      2. VAE latent space refinement
      3. Multi-agent validation (AutoGen)
      4. Self-reflective RAG via LangChain
    """
    if not settings.gemini_api_key:
        raise HTTPException(status_code=400, detail="GEMINI_API_KEY is not configured.")

    property_constraints = {
        "logP": {"min": req.logp_min, "max": req.logp_max},
        "molecularWeight": {"min": req.mw_min, "max": req.mw_max},
        "QED": {"min": req.qed_min, "max": req.qed_max},
    }

    try:
        results = await run_pipeline(
            target_description=req.target_description,
            property_constraints=property_constraints,
            num_molecules=req.num_molecules,
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
