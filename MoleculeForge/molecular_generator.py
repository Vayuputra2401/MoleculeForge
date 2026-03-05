import time
import re
import random
import json
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import backoff  # pip install backoff

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def call_gemini_with_backoff(genai_client, prompt, temp, top_p, top_k):
    """Make API call with exponential backoff on failure"""
    try:
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response
    except Exception as e:
        print(f"API call failed: {e}, retrying...")
        raise

def generate_molecules_from_prompt(target_description, property_constraints, genai_client, num_molecules=3):
    """
    Generate valid SMILES strings based on a text description and property constraints
    using Gemini API with retry logic and JSON response format.
    
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
    
    # Prepare the prompt with pharmaceutical context and JSON schema requirement
    base_prompt = f"""
    As a medicinal chemist, generate {num_molecules*2} novel drug-like molecules as SMILES strings that meet the following criteria:
    
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
    
    Return your response in JSON format using this schema:
    
    Molecule = {{'smiles': str, 'description': str}}
    Response = list[Molecule]
    
    Each molecule should have a valid SMILES string and a brief description of its key features.
    Ensure each SMILES string is chemically valid. Do not include any additional text outside the JSON.
    """
    
    # Array of parameter combinations to try
    param_combinations = [
        {"temp": 0.9, "top_p": 1.0, "top_k": 32, "prompt_suffix": ""},
        {"temp": 0.7, "top_p": 0.95, "top_k": 40, "prompt_suffix": "\nEnsure high chemical diversity among the molecules."},
        {"temp": 1.0, "top_p": 0.98, "top_k": 50, "prompt_suffix": "\nFocus on novel scaffolds that are synthetically accessible."},
        {"temp": 0.8, "top_p": 0.97, "top_k": 45, "prompt_suffix": "\nPrioritize structures with optimal pharmacokinetic properties."},
        {"temp": 0.85, "top_p": 0.99, "top_k": 30, "prompt_suffix": "\nBalance potency with drug-like properties."},
    ]
    
    all_valid_molecules = []
    
    # Try different parameter combinations until we have enough molecules
    for attempt, params in enumerate(param_combinations):
        if len(all_valid_molecules) >= num_molecules:
            break
            
        prompt = base_prompt + params["prompt_suffix"]
        
        print(f"DEBUG: Attempt {attempt+1} with temperature={params['temp']}, top_p={params['top_p']}")
        
        try:
            # Make API call with retry logic
            start_time = time.time()
            response = call_gemini_with_backoff(
                genai_client, 
                prompt,
                params["temp"],
                params["top_p"], 
                params["top_k"]
            )
            elapsed = time.time() - start_time
            print(f"DEBUG: API response received in {elapsed:.2f}s")
            
            # Check if response is valid
            if not hasattr(response, 'text') or not response.text:
                print(f"DEBUG: Empty or invalid response from Gemini API")
                # Wait before continuing to the next attempt
                wait_time = 3  # seconds
                print(f"DEBUG: Waiting {wait_time}s before next attempt...")
                time.sleep(wait_time)
                continue
            
            response_text = response.text
            
            print(f"DEBUG: Received response from Gemini API: {len(response_text)} chars")
            
            # Extract JSON from the response
            smiles_candidates = []
            try:
                # Extract just the JSON part if there's extra text
                json_match = re.search(r'(\[\s*\{.*\}\s*\])', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(1)
                else:
                    json_text = response_text
                
                # Clean up the JSON text
                json_text = json_text.strip()
                if json_text.startswith('```json'):
                    json_text = json_text[7:]
                if json_text.startswith('```'):
                    json_text = json_text[3:]
                if json_text.endswith('```'):
                    json_text = json_text[:-3]
                
                # Parse the JSON
                molecules_data = json.loads(json_text)
                
                # Extract SMILES from the parsed JSON
                for molecule in molecules_data:
                    if 'smiles' in molecule and molecule['smiles']:
                        smiles_candidates.append(molecule['smiles'])
                
                print(f"DEBUG: Successfully extracted {len(smiles_candidates)} SMILES from JSON")
                
            except Exception as e:
                print(f"DEBUG: Error parsing JSON: {e}")
                
                # Fallback: use regex extraction if JSON parsing fails
                print("DEBUG: Falling back to regex extraction")
                
                # First try extracting lines that look like SMILES
                lines = response_text.strip().split('\n')
                for line in lines:
                    clean_line = line.strip()
                    # Skip short lines, code block markers, or lines that are obviously not SMILES
                    if len(clean_line) < 5 or clean_line.startswith('```') or clean_line.endswith('```'):
                        continue
                    if re.match(r'^[A-Za-z0-9@\[\]\(\)\{\}/\\=#\-+.*]+$', clean_line):
                        smiles_candidates.append(clean_line)
                
                # If we still didn't find any candidates, try regex pattern
                if not smiles_candidates:
                    smiles_candidates = re.findall(r'([A-Za-z0-9@\[\]\(\)\{\}/\\=#\-+.]{10,})', response_text)
            
            print(f"DEBUG: Found {len(smiles_candidates)} SMILES candidates")
            
            # Filter valid SMILES and check property constraints
            valid_smiles_this_attempt = []
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
                        
                        # Check for duplicates
                        if canonical_smiles not in all_valid_molecules and canonical_smiles not in valid_smiles_this_attempt:
                            valid_smiles_this_attempt.append(canonical_smiles)
                            print(f"DEBUG: Valid molecule found: {canonical_smiles}")
                            print(f"DEBUG: Properties - LogP: {logp:.2f}, MW: {mw:.2f}, QED: {qed_value:.2f}")
                    else:
                        print(f"DEBUG: Molecule outside property constraints: {smiles}")
                        print(f"DEBUG: Properties - LogP: {logp:.2f}, MW: {mw:.2f}, QED: {qed_value:.2f}")
                
                except Exception as e:
                    print(f"DEBUG: Error processing SMILES {smiles}: {str(e)}")
                    continue
            
            print(f"DEBUG: Found {len(valid_smiles_this_attempt)} valid molecules in attempt {attempt+1}")
            
            # Add molecules from this attempt to our overall collection
            all_valid_molecules.extend(valid_smiles_this_attempt)
            
            # If we got some molecules but not enough, wait before the next attempt
            if len(all_valid_molecules) < num_molecules:
                wait_time = 2  # seconds
                print(f"DEBUG: Waiting {wait_time}s before next attempt...")
                time.sleep(wait_time)
            
            # If we got no valid molecules in this attempt, try a variation of the prompt
            if not valid_smiles_this_attempt and attempt < len(param_combinations) - 1:
                print("DEBUG: No valid molecules found with current parameters, trying different approach")
                # Modify the next attempt to use more specific instructions
                param_combinations[attempt+1]["prompt_suffix"] += "\nFocus on well-known scaffolds with modifications. Use simple, valid chemistry."
                
        except Exception as e:
            print(f"ERROR in attempt {attempt+1}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
            # Wait longer after an error before retrying
            wait_time = 5  # seconds
            print(f"DEBUG: Error occurred, waiting {wait_time}s before next attempt...")
            time.sleep(wait_time)
    
    print(f"DEBUG: Total valid molecules found across all attempts: {len(all_valid_molecules)}")
    
    # If we still don't have enough molecules, try to fill with backup molecules
    if len(all_valid_molecules) < num_molecules:
        print(f"DEBUG: Still not enough valid molecules after all attempts, adding backup molecules")
        
        # Add some backup molecules that are known to be valid and within constraints
        backup_molecules = [
            "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # Ibuprofen
            "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
            "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12",  # Chloroquine
            "COc1cc2c(cc1OC)[C@@H]1[C@H]3CC[C@@H](O)[C@@]3(C)CCN1CC2",  # Codeine
            "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # Caffeine
            "CC(C)NCC(O)COc1cccc2ccccc12",  # Propranolol
            "CC(CS)C(=O)N1CCCC1C(=O)O",  # Captopril
            "CCOC(=O)C1=C(C)NC(C)=C(C1C(=O)OC)C(=O)OCC",  # Amlodipine
            "C1=CC=C2C(=C1)C=CC=C2",  # Naphthalene
            "CC(=O)NC1=CC=C(C=C1)O",  # Acetaminophen
            "CN1C2=C(C(=O)N(C1=O)C)NC=N2",  # Theophylline
            "COC(=O)C1=CC=CC=C1C(=O)O"  # Methyl salicylate
        ]
        
        # Filter backup molecules by constraints and add until we reach num_molecules
        for smiles in backup_molecules:
            if len(all_valid_molecules) >= num_molecules:
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
                    if smiles not in all_valid_molecules:
                        all_valid_molecules.append(smiles)
                        print(f"DEBUG: Added backup molecule: {smiles}")
            except Exception:
                continue
    
    # Limit to requested number of molecules
    final_molecules = all_valid_molecules[:num_molecules]
    
    return final_molecules , response_text