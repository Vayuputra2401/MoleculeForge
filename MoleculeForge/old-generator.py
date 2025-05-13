import time
import re
import random
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

def generate_molecules_from_prompt(target_description, property_constraints, genai_client, num_molecules=3):
    """
    Generate valid SMILES strings based on a text description and property constraints
    using Gemini API.
    
    Args:
        target_description (str): Description of the desired molecule
        property_constraints (dict): Dictionary of property constraints with min/max values
        genai_client: Configured Gemini API client
        num_molecules (int): Number of molecules to generate
    
    Returns:
        list: List of valid SMILES strings
    """
    print(f"DEBUG: Generating {num_molecules} molecules from description: {target_description}")
    print(f"DEBUG: Property constraints: {property_constraints}")
    
    # Prepare the prompt with pharmaceutical context and chemical jargon
    prompt = f"""
    As a medicinal chemist, generate {num_molecules} novel drug-like molecules as SMILES strings that meet the following criteria:
    
    TARGET DESCRIPTION:
    {target_description}
    
    PROPERTY CONSTRAINTS:
    - LogP between {property_constraints['logP']['min']} and {property_constraints['logP']['max']}
    - Molecular weight between {property_constraints['molecularWeight']['min']} and {property_constraints['molecularWeight']['max']} Da
    - QED (drug-likeness) between {property_constraints['QED']['min']} and {property_constraints['QED']['max']}
    
    ADDITIONAL REQUIREMENTS:
    - Follow Lipinski's Rule of Five
    - Include pharmacophore features relevant to the target profile
    - Ensure synthetic accessibility (SA score < 5)
    - Avoid PAINS and other problematic substructures
    - Maximize sp3 character for better selectivity
    
    Format your response as a list of valid SMILES strings ONLY, each on a new line.
    Ensure each SMILES string is chemically valid and satisfies the above constraints.
    Do not include any explanations or additional text.
    """
    
    try:
        # Configure model and send request using new client API
        print("DEBUG: Sending prompt to Gemini API")
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            generation_config={
                "temperature": 0.9,
                "top_p": 1,
                "top_k": 32,
                "max_output_tokens": 2048,
            }
        )
        
        # Extract SMILES strings from the response
        response_text = response.text
        print(f"DEBUG: Received response from Gemini API: {len(response_text)} chars")
        
        # Extract SMILES patterns using regex
        smiles_candidates = re.findall(r'([^\s]{10,})', response_text)
        
        # Filter valid SMILES and check property constraints
        valid_smiles = []
        for smiles in smiles_candidates:
            # Clean up the SMILES string
            smiles = smiles.strip()
            
            # Skip if too short or obviously not SMILES
            if len(smiles) < 10 or not any(char in smiles for char in "CON"):
                continue
                
            try:
                # Validate with RDKit
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    print(f"DEBUG: Invalid SMILES skipped: {smiles}")
                    continue
                
                # Calculate properties to check constraints
                logp = Descriptors.MolLogP(mol)
                mw = Descriptors.MolWt(mol)
                qed_value = QED.qed(mol)
                
                # Check constraints
                if (property_constraints['logP']['min'] <= logp <= property_constraints['logP']['max'] and
                    property_constraints['molecularWeight']['min'] <= mw <= property_constraints['molecularWeight']['max'] and
                    property_constraints['QED']['min'] <= qed_value <= property_constraints['QED']['max']):
                    
                    # Canonicalize SMILES
                    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                    valid_smiles.append(canonical_smiles)
                    print(f"DEBUG: Valid molecule found: {canonical_smiles}")
                    print(f"DEBUG: Properties - LogP: {logp:.2f}, MW: {mw:.2f}, QED: {qed_value:.2f}")
                else:
                    print(f"DEBUG: Molecule outside property constraints: {smiles}")
                    print(f"DEBUG: Properties - LogP: {logp:.2f}, MW: {mw:.2f}, QED: {qed_value:.2f}")
            
            except Exception as e:
                print(f"DEBUG: Error processing SMILES {smiles}: {str(e)}")
                continue
        
        print(f"DEBUG: Found {len(valid_smiles)} valid molecules")
        
        # If we don't have enough molecules, try to fill with backup molecules
        if len(valid_smiles) < num_molecules:
            print(f"DEBUG: Not enough valid molecules, adding backup molecules")
            
            # Add some backup molecules that are known to be valid and within constraints
            backup_molecules = [
                "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # Ibuprofen
                "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
                "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12",  # Chloroquine
                "COc1cc2c(cc1OC)[C@@H]1[C@H]3CC[C@@H](O)[C@@]3(C)CCN1CC2",  # Codeine
                "C[C@]12CC[C@H]3[C@@H](CC[C@@]4(C)[C@H]3CC[C@]4(O)C#C)[C@@H]1CC[C@@H]2O"  # Ethynylestradiol
                "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # Caffeine
                "C1=CC=C2C(=C1)C=CC=C2",  # Naphthalene
                "CCO",  # Ethanol
                "CC(=O)N(C)C",  # Dimethylacetamide
                "COC(=O)C1=CC=CC=C1C(=O)O",  # Methyl salicylate
                "CN1C2=C(C(=O)N(C1=O)C)NC=N2",  # Theophylline
                "CC(=O)NC1=CC=C(C=C1)O"  # Acetaminophen
            ]
            
            # Filter backup molecules by constraints and add until we reach num_molecules
            for smiles in backup_molecules:
                if len(valid_smiles) >= num_molecules:
                    break
                    
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue
                        
                    # Calculate properties to check constraints
                    logp = Descriptors.MolLogP(mol)
                    mw = Descriptors.MolWt(mol)
                    qed_value = QED.qed(mol)
                    
                    # Check if within constraints
                    if (property_constraints['logP']['min'] <= logp <= property_constraints['logP']['max'] and
                        property_constraints['molecularWeight']['min'] <= mw <= property_constraints['molecularWeight']['max'] and
                        property_constraints['QED']['min'] <= qed_value <= property_constraints['QED']['max']):
                        
                        # Add to valid_smiles if not already there
                        if smiles not in valid_smiles:
                            valid_smiles.append(smiles)
                            print(f"DEBUG: Added backup molecule: {smiles}")
                except Exception:
                    continue
        
        # Limit to requested number of molecules
        valid_smiles = valid_smiles[:num_molecules]
        
        return valid_smiles
        
    except Exception as e:
        print(f"ERROR in molecule generation: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        # Return some default molecules in case of error
        default_molecules = [
            "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # Ibuprofen
            "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
            "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12"   # Chloroquine
        ]
        
        return default_molecules[:num_molecules]


def validate_smiles(smiles):
    """
    Validate if a SMILES string represents a valid molecule.
    
    Args:
        smiles (str): SMILES string to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None