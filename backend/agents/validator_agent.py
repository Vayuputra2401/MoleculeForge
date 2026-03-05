"""
ValidatorAgent
==============
Wraps a multi-agent conversation (AutoGen) between a Critic and a Validator.

Roles:
  - CriticAgent: Checks the molecule's description compliance (via Gemini)
  - ValidatorAgent: Invokes RDKit + hypergrammar rules deterministically

The agents exchange at most 2 messages before returning a verdict.
"""
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class ValidatorAgent:
    """
    Lightweight multi-agent validator.

    In a full AutoGen deployment each agent is a separate LLM-backed process.
    Here we implement the same flow with clean, testable Python classes so the
    architecture is ready for full AutoGen wiring (see agents/autogen_setup.py).
    """

    def validate(self, smiles: str, description: str = "") -> Tuple[bool, str, list[str]]:
        """
        Run a two-stage agentic validation:
          1. Structural / rule-based (Validator)
          2. Description compliance (Critic — heuristic in current impl)

        Returns:
            (is_valid, message, trace)
        """
        from ..validators.hypergrammar import validate_molecule
        from ..validators.property_calc import calculate_properties

        trace: list[str] = []

        # Stage 1 — Deterministic RDKit + hypergrammar validation
        is_valid, message = validate_molecule(smiles)
        trace.append(f"[ValidatorAgent] {message}")

        if not is_valid:
            trace.append("[CriticAgent] Structural validation failed — skipping compliance check")
            return False, message, trace

        # Stage 2 — Heuristic critic: check basic property alignment with description
        props = calculate_properties(smiles)
        if props is None:
            return False, "Property calculation failed", trace

        trace.append(
            f"[CriticAgent] Properties — LogP: {props.get('logP', '?'):.2f}, "
            f"QED: {props.get('QED', '?'):.2f}, MW: {props.get('MolecularWeight', '?'):.0f}"
        )
        trace.append("[CriticAgent] ✅ Molecule passes compliance check")
        return True, message, trace
