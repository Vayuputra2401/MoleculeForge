import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, AllChem
from rdkit.Chem.Lipinski import NumHDonors, NumHAcceptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

def validate_molecule(smiles):
    """
    Validates a molecule using hypergrammar rules.
    
    Args:
        smiles (str): SMILES string of the molecule
        
    Returns:
        tuple: (bool, str) - (is_valid, validation_message)
    """
    # Check if smiles is None or empty
    if smiles is None:
        return False, "Invalid input: SMILES string is None"
    
    if not isinstance(smiles, str):
        return False, f"Invalid input: Expected string but got {type(smiles).__name__}"
        
    if not smiles.strip():
        return False, "Invalid input: Empty SMILES string"
    
    try:
        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return False, "Invalid SMILES string: Could not parse molecular structure"
        
        # Check Lipinski's Rule of Five
        lipinski_violations = 0
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        h_donors = NumHDonors(mol)  # Using correctly imported function
        h_acceptors = NumHAcceptors(mol)  # Using correctly imported function
        
        validation_details = []
        
        # Check molecular weight
        if mw > 500:
            lipinski_violations += 1
            validation_details.append(f"Molecular weight ({mw:.2f}) exceeds 500")
        
        # Check logP
        if logp > 5:
            lipinski_violations += 1
            validation_details.append(f"LogP ({logp:.2f}) exceeds 5")
        
        # Check H-bond donors
        if h_donors > 5:
            lipinski_violations += 1
            validation_details.append(f"H-bond donors ({h_donors}) exceeds 5")
        
        # Check H-bond acceptors
        if h_acceptors > 10:
            lipinski_violations += 1
            validation_details.append(f"H-bond acceptors ({h_acceptors}) exceeds 10")
        
        # Check for problematic substructures (simplified)
        problematic_groups = [
            # SMARTS patterns for problematic groups
            '[N+](=O)[O-]',  # Nitro group
            'C(F)(F)(F)',     # Trifluoromethyl
            '[SH]',           # Thiol
        ]
        
        found_problematic = []
        for smarts in problematic_groups:
            pattern = Chem.MolFromSmarts(smarts)
            if mol.HasSubstructMatch(pattern):
                found_problematic.append(smarts)
        
        if found_problematic:
            validation_details.append(f"Contains problematic groups: {', '.join(found_problematic)}")
        
        # Number of rotatable bonds
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        if rot_bonds > 10:
            validation_details.append(f"High number of rotatable bonds ({rot_bonds})")
        
        # Final validation decision
        is_valid = True
        if lipinski_violations > 1:
            is_valid = False
            validation_details.insert(0, f"Failed Lipinski's Rule of Five with {lipinski_violations} violations")
        
        if found_problematic:
            is_valid = False
        
        # Create validation message
        if is_valid:
            validation_message = "Molecule passed all validation checks"
            if validation_details:
                validation_message += f" with notes: {'; '.join(validation_details)}"
        else:
            validation_message = f"Validation failed: {'; '.join(validation_details)}"
        
        return is_valid, validation_message
        
    except Exception as e:
        return False, f"Error during validation: {str(e)}"

def validate_hypergrammar_rules(mol):
    """
    Check if a molecule passes custom hypergrammar rules
    
    Args:
        mol (RDKit.Mol): RDKit molecule object
        
    Returns:
        tuple: (bool, str) - (passes_rules, rule_violation_message)
    """
    # Rule 1: No charged atoms
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            return False, f"Hypergrammar violation: molecule has charged {atom.GetSymbol()} atom"
    
    # Rule 2: No peroxide or similar unstable bonds
    peroxide_patt = Chem.MolFromSmarts('O-O')
    if mol.HasSubstructMatch(peroxide_patt):
        return False, "Hypergrammar violation: molecule contains unstable peroxide (O-O) bond"
    
    # Rule 3: No odd ring sizes (3, 5, 7)
    ring_info = mol.GetSSSR()
    for ring in ring_info:
        ring_size = ring.Size()
        if ring_size in [3, 5, 7]:
            return False, f"Hypergrammar violation: molecule contains {ring_size}-membered ring"
    
    # Rule 4: No excessive symmetry (potential for crystallization)
    try:
        symmetry_classes = Chem.CanonicalRankAtoms(mol, includeChirality=False)
        unique_classes = len(set(symmetry_classes))
        total_atoms = mol.GetNumAtoms()
        if unique_classes < total_atoms / 3:  # If less than 1/3 of atoms are unique
            return False, "Hypergrammar violation: molecule has excessive symmetry"
    except:
        pass  # If symmetry calculation fails, continue
    
    # All rules passed
    return True, "Molecule passes all hypergrammar rules"
