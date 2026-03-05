"""
Hypergrammar Validator (refactored from legacy hypergrammar_validator.py)
Validates a SMILES string against chemical feasibility and drug-likeness rules.
"""
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit.Chem.Lipinski import NumHDonors, NumHAcceptors
from typing import Tuple

_PROBLEMATIC_SMARTS = [
    ("[N+](=O)[O-]",  "Nitro group"),
    ("C(F)(F)(F)",    "Trifluoromethyl"),
    ("[SH]",          "Free thiol"),
    ("O-O",           "Peroxide bond"),
]


def validate_molecule(smiles: str) -> Tuple[bool, str]:
    """
    Validate a SMILES string using Lipinski Ro5 and hypergrammar rules.

    Returns:
        (is_valid, message)
    """
    if not smiles or not isinstance(smiles, str) or not smiles.strip():
        return False, "Invalid or empty SMILES input"

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "Cannot parse SMILES — invalid structure"

    notes: list[str] = []
    violations = 0

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = NumHDonors(mol)
    hba = NumHAcceptors(mol)

    if mw > 500:
        violations += 1
        notes.append(f"MW {mw:.0f} > 500")
    if logp > 5:
        violations += 1
        notes.append(f"LogP {logp:.2f} > 5")
    if hbd > 5:
        violations += 1
        notes.append(f"HBD {hbd} > 5")
    if hba > 10:
        violations += 1
        notes.append(f"HBA {hba} > 10")

    if violations > 1:
        return False, f"Lipinski failed ({violations} violations): {'; '.join(notes)}"

    # Problematic substructure scan
    for smarts, label in _PROBLEMATIC_SMARTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt and mol.HasSubstructMatch(patt):
            return False, f"Problematic substructure: {label}"

    rot = Descriptors.NumRotatableBonds(mol)
    if rot > 10:
        notes.append(f"High rot bonds ({rot})")

    msg = "Passed all validation checks"
    if notes:
        msg += f" — notes: {'; '.join(notes)}"
    return True, msg
