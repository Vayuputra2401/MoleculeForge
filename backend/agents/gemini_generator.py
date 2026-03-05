"""
GeminiGeneratorAgent
====================
Uses the fine-tuned Gemini model with Chain-of-Thought prompting to generate
initial SMILES candidates from a natural language molecular description.

Chain-of-Thought strategy:
  Prompt Gemini to first reason about the desired pharmacophore, scaffold, and
  physicochemical profile, THEN emit SMILES strings. This CoT trace is captured
  and returned alongside the candidates.
"""
import json
import re
import asyncio
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

_COT_PROMPT_TEMPLATE = """
You are an expert medicinal chemist.

### Task
Generate {n} novel, drug-like SMILES that satisfy the constraints below.

### Target Description
{description}

### Property Constraints
- LogP: {logp_min} – {logp_max}
- Molecular Weight: {mw_min} – {mw_max} Da
- QED (drug-likeness): {qed_min} – {qed_max}

### Additional Rules
- Follow Lipinski's Rule of Five
- Avoid PAINS and toxic substructures
- SA score < 5 (synthetically accessible)
- Maximise sp3 character for selectivity

### Instructions
FIRST, reason step-by-step about the ideal scaffold, key pharmacophores, and how
to satisfy the constraints simultaneously. Label this section ## Reasoning.

THEN emit a JSON array labelled ## Molecules in this exact schema:
[
  {{"smiles": "<SMILES>", "rationale": "<one-line rationale>"}},
  ...
]
Do not emit any text after the JSON array.
"""


class GeminiGeneratorAgent:
    def __init__(self, genai_client, model: str = "gemini-2.0-flash"):
        self.client = genai_client
        self.model = model

    async def generate(
        self,
        target_description: str,
        property_constraints: dict,
        num_molecules: int = 6,
    ) -> Tuple[list[str], list[str]]:
        """
        Calls Gemini with Chain-of-Thought prompting.

        Returns:
            (smiles_list, cot_trace) — list of SMILES and CoT log lines
        """
        prompt = _COT_PROMPT_TEMPLATE.format(
            n=num_molecules,
            description=target_description,
            logp_min=property_constraints["logP"]["min"],
            logp_max=property_constraints["logP"]["max"],
            mw_min=property_constraints["molecularWeight"]["min"],
            mw_max=property_constraints["molecularWeight"]["max"],
            qed_min=property_constraints["QED"]["min"],
            qed_max=property_constraints["QED"]["max"],
        )

        logger.info("[GeminiAgent] Sending CoT prompt (%d chars)", len(prompt))

        # Run blocking SDK call in a thread pool so it doesn't block the event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.model, contents=prompt
            ),
        )

        response_text = response.text or ""
        cot_trace = self._extract_reasoning(response_text)
        smiles_list = self._extract_smiles(response_text)

        logger.info(
            "[GeminiAgent] Got %d candidates; CoT trace: %d lines",
            len(smiles_list),
            len(cot_trace),
        )
        return smiles_list, cot_trace

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_reasoning(self, text: str) -> list[str]:
        """Pull out the ## Reasoning section as individual lines."""
        match = re.search(r"##\s*Reasoning(.+?)##\s*Molecules", text, re.DOTALL | re.IGNORECASE)
        if match:
            lines = [l.strip() for l in match.group(1).strip().splitlines() if l.strip()]
            return [f"[CoT] {l}" for l in lines]
        return []

    def _extract_smiles(self, text: str) -> list[str]:
        """Parse the JSON array from the ## Molecules section."""
        # Try structured JSON first
        json_match = re.search(r"\[.*\]", text, re.DOTALL)
        if json_match:
            try:
                mols = json.loads(json_match.group(0))
                return [m["smiles"] for m in mols if "smiles" in m]
            except Exception:
                pass

        # Fallback: regex for SMILES-like tokens
        return re.findall(r"[A-Za-z0-9@\[\]\(\)\{\}/\\=#\-+.]{10,}", text)
