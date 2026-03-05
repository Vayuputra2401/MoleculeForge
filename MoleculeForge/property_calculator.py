from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Lipinski, Crippen, MolSurf, rdMolDescriptors
import numpy as np
from rdkit.Chem import AllChem, Draw

def calculate_properties(smiles):
    """
    Calculate important molecular properties for a given SMILES string
    
    Args:
        smiles (str): SMILES string of the molecule
        
    Returns:
        dict: Dictionary of calculated properties
    """
    # Parse molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Create properties dictionary
    properties = {}
    
    # Basic properties
    properties['SMILES'] = smiles
    properties['MolecularFormula'] = Chem.rdMolDescriptors.CalcMolFormula(mol)
    properties['MolecularWeight'] = Descriptors.MolWt(mol)
    properties['NumAtoms'] = mol.GetNumAtoms()
    properties['NumBonds'] = mol.GetNumBonds()
    properties['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
    properties['NumHBondDonors'] = Descriptors.NumHDonors(mol)
    properties['NumHBondAcceptors'] = Descriptors.NumHAcceptors(mol)
    
    # Physical properties
    properties['logP'] = Descriptors.MolLogP(mol)
    properties['TPSA'] = Descriptors.TPSA(mol)
    properties['LabuteASA'] = Descriptors.LabuteASA(mol)
    
    # Ring information
    properties['NumRings'] = Descriptors.RingCount(mol)
    properties['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
    properties['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
    
    # Drug-likeness and medicinal chemistry properties
    properties['QED'] = QED.qed(mol)
    properties['FractionCSP3'] = Descriptors.FractionCSP3(mol)
    
    # Lipinski's Rule of 5 violations
    violations = 0
    if properties['MolecularWeight'] > 500:
        violations += 1
    if properties['logP'] > 5:
        violations += 1
    if properties['NumHBondDonors'] > 5:
        violations += 1
    if properties['NumHBondAcceptors'] > 10:
        violations += 1
    properties['LipinskiViolations'] = violations
    
    # Veber rules (GSK)
    veber_violations = 0
    if properties['NumRotatableBonds'] > 10:
        veber_violations += 1
    if properties['TPSA'] > 140:
        veber_violations += 1
    properties['VeberViolations'] = veber_violations
    
    # Synthetic accessibility
    try:
        properties['SyntheticAccessibility'] = rdMolDescriptors.CalcSyntheticAccessibilityScore(mol)
    except:
        properties['SyntheticAccessibility'] = None
    
    # Natural product-likeness
    try:
        properties['NPLikeness'] = rdMolDescriptors.CalcNPScore(mol)
    except:
        properties['NPLikeness'] = None
    
    # Water solubility estimation (log S)
    # Using a simple model based on logP and MW
    try:
        logS = 0.8 - properties['logP'] - 0.01 * properties['MolecularWeight'] / 100
        properties['EstimatedLogSolubility'] = logS
    except:
        properties['EstimatedLogSolubility'] = None
    
    return properties

def calculate_fingerprint_properties(smiles):
    """
    Calculate molecular fingerprint-based properties
    
    Args:
        smiles (str): SMILES string of the molecule
        
    Returns:
        dict: Dictionary of fingerprint-based properties
    """
    # Parse molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Properties dictionary
    fp_properties = {}
    
    # Morgan (ECFP) fingerprint
    from rdkit.Chem import AllChem
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    fp_properties['MorganFP'] = morgan_fp
    
    # Calculate number of bits set (feature count)
    fp_properties['BitCount'] = sum(morgan_fp.GetOnBits())
    fp_properties['BitDensity'] = fp_properties['BitCount'] / 2048
    
    return fp_properties

def calculate_similarity(smiles1, smiles2, fp_type='morgan'):
    """
    Calculate similarity between two molecules
    
    Args:
        smiles1 (str): SMILES string of first molecule
        smiles2 (str): SMILES string of second molecule
        fp_type (str): Type of fingerprint to use ('morgan', 'maccs', 'rdkit')
        
    Returns:
        float: Tanimoto similarity between molecules
    """
    # Parse molecules
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return None
    
    # Calculate fingerprints and similarity
    from rdkit import DataStructs
    
    if fp_type == 'morgan':
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 2048)
    elif fp_type == 'maccs':
        fp1 = AllChem.GetMACCSKeysFingerprint(mol1)
        fp2 = AllChem.GetMACCSKeysFingerprint(mol2)
    else:  # rdkit
        fp1 = Chem.RDKFingerprint(mol1)
        fp2 = Chem.RDKFingerprint(mol2)
    
    # Calculate Tanimoto similarity
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    return similarity
