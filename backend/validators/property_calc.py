"""
Property Calculator (refactored from legacy property_calculator.py)
Calculates RDKit molecular properties for display and scoring.
"""
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors
from typing import Optional

def calculate_properties(smiles: str) -> Optional[dict]:
    """
    Calculate key physicochemical properties for a SMILES string.
    Returns None if the molecule cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    props: dict = {}
    props["SMILES"]             = smiles
    props["MolecularFormula"]   = rdMolDescriptors.CalcMolFormula(mol)
    props["MolecularWeight"]    = round(Descriptors.MolWt(mol), 2)
    props["logP"]               = round(Descriptors.MolLogP(mol), 3)
    props["QED"]                = round(QED.qed(mol), 3)
    props["TPSA"]               = round(Descriptors.TPSA(mol), 2)
    props["NumHBondDonors"]     = Descriptors.NumHDonors(mol)
    props["NumHBondAcceptors"]  = Descriptors.NumHAcceptors(mol)
    props["NumRotatableBonds"]  = Descriptors.NumRotatableBonds(mol)
    props["NumRings"]           = Descriptors.RingCount(mol)
    props["NumAromaticRings"]   = Descriptors.NumAromaticRings(mol)
    props["FractionCSP3"]       = round(Descriptors.FractionCSP3(mol), 3)
    props["LabuteASA"]          = round(Descriptors.LabuteASA(mol), 2)

    violations = sum([
        props["MolecularWeight"] > 500,
        props["logP"] > 5,
        props["NumHBondDonors"] > 5,
        props["NumHBondAcceptors"] > 10,
    ])
    props["LipinskiViolations"] = violations

    try:
        props["SyntheticAccessibility"] = round(rdMolDescriptors.CalcSyntheticAccessibilityScore(mol), 2)
    except Exception:
        props["SyntheticAccessibility"] = None

    # Estimated water solubility (Yalkowsky)
    props["EstimatedLogS"] = round(0.8 - props["logP"] - 0.01 * props["MolecularWeight"] / 100, 3)

    return props
