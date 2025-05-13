import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED
from rdkit import RDLogger
import os
import random
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from sklearn.decomposition import PCA

# Import VAE
from molecule_vae import MoleculeVAE

# Suppress RDKit logging
RDLogger.DisableLog('rdApp.*')

class MoleculeGenerator:
    """
    Molecule generation and refinement using a pre-trained VAE model
    with SR-guided latent space exploration
    """
    def __init__(self):
        """Initialize the molecule generator with pre-trained VAE"""
        print("DEBUG: Initializing MoleculeGenerator with VAE")
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DEBUG: Using device: {self.device}")
        
        # Load pre-trained VAE model
        self.vae = MoleculeVAE(device=self.device)
        
        # Load reference molecules for latent space exploration
        self.reference_molecules = self._load_reference_molecules()
        self.reference_embeddings = {}
        
        # Cache for molecule embeddings from the VAE
        self.embedding_cache = {}
        
        # Initialize SR models for property prediction
        self.sr_models = {
            'logP': None,
            'QED': None,
            'MolecularWeight': None
        }
        
        # Initialize PCA for dimensionality reduction
        self.pca = None
        
        # Train initial SR models with reference data
        self._train_initial_sr_models()
        
        print("DEBUG: MoleculeGenerator initialized")
    
    def _load_reference_molecules(self):
        """Load reference molecules for latent space exploration"""
        print("DEBUG: Loading reference molecules")
        
        # In a real implementation, this would load from a database
        # For this prototype, we'll use a predefined list
        reference_molecules = [
            "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # Ibuprofen
            "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
            "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12",  # Chloroquine
            "COc1cc2c(cc1OC)[C@@H]1[C@H]3CC[C@@H](O)[C@@]3(C)CCN1CC2",  # Codeine
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            "CC(C)NCC(O)COc1cccc2ccccc12",  # Propranolol
            "CC(CS)C(=O)N1CCCC1C(=O)O",  # Captopril
            "CCOC(=O)C1=C(C)NC(C)=C(C1C(=O)OC)C(=O)OCC",  # Amlodipine
            "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",  # Atenolol
            "CC(C)C(=O)Nc1ccc(Cl)c(c1)C(=O)c1ccc(F)cc1",  # Flurbiprofen
            "CC(C)C(=O)Nc1ccc(Cl)c(c1)C(=O)c1ccc(OC)cc1",  # Naproxen
            "CC(=O)Nc1ccc(O)cc1",  # Acetaminophen
            "c1ccc2c(c1)ccc3c2ccc4c5ccccc5ccc43",  # Pentacene
            "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12",  # Chloroquine
            "COc1cc(OC)c2C(=O)c3ccc(OC)c(OC)c3C(=O)c2c1OC",  # Tetramethoxyflavone
            "CC(C)C(=O)O",  # Isobutyric acid
            "C=C(C)C(=O)OCCC",  # Propyl methacrylate
            "CC(C)(C)C(=O)O",  # Pivalic acid
        ]
        
        return reference_molecules

    def get_molecule_embedding(self, smiles):
        """
        Get embedding for a molecule using the VAE encoder
        
        Args:
            smiles (str): SMILES string
            
        Returns:
            numpy.ndarray: Embedding vector from VAE latent space
        """
        # Check if embedding is cached
        if smiles in self.embedding_cache:
            return self.embedding_cache[smiles]
        
        try:
            # Use the VAE encoder to get the latent representation
            z_mean, z_log_var = self.vae.encode(smiles)
            
            # Use the mean as the embedding
            embedding = z_mean.cpu().detach().numpy()[0]
            
            # Cache embedding
            self.embedding_cache[smiles] = embedding
            
            return embedding
        
        except Exception as e:
            print(f"DEBUG: Error getting VAE embedding for {smiles}: {str(e)}")
            # Generate a random embedding in latent space as fallback
            embedding = np.random.normal(0, 1, self.vae.latent_dim)
            self.embedding_cache[smiles] = embedding
            return embedding
    
    def _train_initial_sr_models(self):
        """Train initial SR models with reference molecules"""
        print("DEBUG: Training initial Symbolic Regression models")
        
        # Get properties and embeddings for reference molecules
        embeddings = []
        logp_values = []
        qed_values = []
        mw_values = []
        
        for smiles in self.reference_molecules:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                    
                # Calculate properties
                logp = Descriptors.MolLogP(mol)
                qed = QED.qed(mol)
                mw = Descriptors.MolWt(mol)
                
                # Get embedding from VAE
                embedding = self.get_molecule_embedding(smiles)
                
                # Collect data
                embeddings.append(embedding)
                logp_values.append(logp)
                qed_values.append(qed)
                mw_values.append(mw)
                
            except Exception as e:
                print(f"Error processing reference molecule {smiles}: {e}")
                continue
        
        if len(embeddings) < 5:
            print("Not enough reference molecules for SR training")
            return
        
        # Convert to arrays
        X = np.array(embeddings)
        y_logp = np.array(logp_values)
        y_qed = np.array(qed_values)
        y_mw = np.array(mw_values)
        
        # Use PCA to reduce dimensionality for SR
        self.pca = PCA(n_components=min(10, X.shape[0] - 1))
        X_reduced = self.pca.fit_transform(X)
        
        print(f"DEBUG: Reduced VAE embedding dimensions from {X.shape[1]} to {X_reduced.shape[1]}")
        
        # Define SR models with appropriate complexity
        try:
            print("DEBUG: Training LogP SR model")
            self.sr_models['logP'] = SymbolicRegressor(
                population_size=300,
                generations=20,
                stopping_criteria=0.01,
                p_crossover=0.7,
                p_subtree_mutation=0.1,
                p_hoist_mutation=0.05,
                p_point_mutation=0.1,
                max_samples=0.9,
                parsimony_coefficient=0.01,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            self.sr_models['logP'].fit(X_reduced, y_logp)
            
            print("DEBUG: Training QED SR model")
            self.sr_models['QED'] = SymbolicRegressor(
                population_size=300,
                generations=20,
                stopping_criteria=0.01,
                p_crossover=0.7,
                p_subtree_mutation=0.1,
                p_hoist_mutation=0.05,
                p_point_mutation=0.1,
                max_samples=0.9,
                parsimony_coefficient=0.01,
                random_state=43,
                n_jobs=-1,
                verbose=0
            )
            self.sr_models['QED'].fit(X_reduced, y_qed)
            
            print("DEBUG: Training MolecularWeight SR model")
            self.sr_models['MolecularWeight'] = SymbolicRegressor(
                population_size=300,
                generations=20,
                stopping_criteria=0.01,
                p_crossover=0.7,
                p_subtree_mutation=0.1,
                p_hoist_mutation=0.05,
                p_point_mutation=0.1,
                max_samples=0.9,
                parsimony_coefficient=0.01,
                random_state=44,
                n_jobs=-1,
                verbose=0
            )
            self.sr_models['MolecularWeight'].fit(X_reduced, y_mw)
            
            # Print model scores
            print(f"DEBUG: LogP SR model score: {self.sr_models['logP'].score(X_reduced, y_logp):.4f}")
            print(f"DEBUG: QED SR model score: {self.sr_models['QED'].score(X_reduced, y_qed):.4f}")
            print(f"DEBUG: MW SR model score: {self.sr_models['MolecularWeight'].score(X_reduced, y_mw):.4f}")
            
        except Exception as e:
            print(f"DEBUG: Error training SR models: {e}")
    
    def decode_from_latent(self, z):
        """
        Decode a latent vector to a SMILES string using the VAE decoder
        
        Args:
            z (numpy.ndarray): Latent vector
            
        Returns:
            str: SMILES string
        """
        try:
            # Convert to tensor and send to device
            z_tensor = torch.tensor(z, dtype=torch.float32).to(self.device).unsqueeze(0)
            
            # Decode using VAE
            decoded_smiles = self.vae.decode(z_tensor)
            
            # Validate the decoded SMILES
            mol = Chem.MolFromSmiles(decoded_smiles)
            if mol is None:
                print("DEBUG: VAE decoded an invalid molecule")
                return None
                
            # Return canonical SMILES
            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            return canonical_smiles
            
        except Exception as e:
            print(f"DEBUG: Error decoding from latent space: {str(e)}")
            return None
    
    def get_reference_embeddings(self, property_name, property_value):
        """
        Get embeddings for reference molecules filtered by property
        
        Args:
            property_name (str): Property name ('logP', 'QED', etc.)
            property_value (float): Target property value
            
        Returns:
            tuple: (list of SMILES, numpy.ndarray of embeddings)
        """
        # Key for caching
        cache_key = f"{property_name}_{property_value:.2f}"
        
        # Check if already computed
        if cache_key in self.reference_embeddings:
            return self.reference_embeddings[cache_key]
        
        # Filter molecules by property
        filtered_smiles = []
        for smiles in self.reference_molecules:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                    
                # Calculate property
                if property_name == 'logP':
                    value = Descriptors.MolLogP(mol)
                elif property_name == 'QED':
                    value = QED.qed(mol)
                elif property_name == 'MolecularWeight':
                    value = Descriptors.MolWt(mol)
                else:
                    continue
                    
                # Check if property is close to target
                if abs(value - property_value) < 2.0:
                    filtered_smiles.append(smiles)
            except:
                continue
        
        # If no molecules match, use all reference molecules
        if not filtered_smiles:
            filtered_smiles = self.reference_molecules
        
        # Get embeddings
        embeddings = []
        for smiles in filtered_smiles:
            embedding = self.get_molecule_embedding(smiles)
            embeddings.append(embedding)
        
        # Cache results
        self.reference_embeddings[cache_key] = (filtered_smiles, np.array(embeddings))
        
        return filtered_smiles, np.array(embeddings)
    
    def predict_properties_with_sr(self, embedding):
        """
        Predict molecular properties using trained SR models
        
        Args:
            embedding (numpy.ndarray): Molecule embedding
            
        Returns:
            dict: Predicted properties
        """
        if self.pca is None or any(model is None for model in self.sr_models.values()):
            return {
                'logP': 0.0,
                'QED': 0.5,
                'MolecularWeight': 300.0
            }
            
        try:
            # Reduce dimensionality
            embedding_reduced = self.pca.transform(embedding.reshape(1, -1))[0]
            
            # Predict properties
            logp_pred = self.sr_models['logP'].predict([embedding_reduced])[0]
            qed_pred = self.sr_models['QED'].predict([embedding_reduced])[0]
            mw_pred = self.sr_models['MolecularWeight'].predict([embedding_reduced])[0]
            
            # Ensure realistic values
            qed_pred = max(0, min(1, qed_pred))  # QED is between 0 and 1
            mw_pred = max(50, min(1000, mw_pred))  # Reasonable MW range
            
            return {
                'logP': logp_pred,
                'QED': qed_pred,
                'MolecularWeight': mw_pred
            }
        except Exception as e:
            print(f"Error in SR prediction: {e}")
            return {
                'logP': 0.0,
                'QED': 0.5,
                'MolecularWeight': 300.0
            }
    
    def refine_molecule(self, smiles, property_constraints, iterations=20):
        """
        Refine a molecule using VAE and SR-guided latent space exploration
        
        Args:
            smiles (str): Initial SMILES string
            property_constraints (dict): Dictionary with property constraints
            iterations (int): Number of optimization iterations
            
        Returns:
            str: Optimized SMILES string
        """
        # Check for None or invalid input
        if smiles is None or not isinstance(smiles, str) or not smiles.strip():
            print(f"Warning: Invalid input SMILES received in refinement: {smiles}")
            return "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin as fallback
        
        # Parse initial molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Could not parse SMILES in refinement: {smiles}")
            return "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin as fallback
        
        print(f"Refining molecule: {smiles}")
        
        # Calculate initial properties
        initial_logp = Descriptors.MolLogP(mol)
        initial_qed = QED.qed(mol)
        initial_mw = Descriptors.MolWt(mol)
        
        print(f"Initial properties - LogP: {initial_logp:.2f}, QED: {initial_qed:.2f}, MW: {initial_mw:.2f}")
        
        # Get target property values (midpoint of ranges)
        target_logp = (property_constraints['logP']['min'] + property_constraints['logP']['max']) / 2
        target_qed = (property_constraints['QED']['min'] + property_constraints['QED']['max']) / 2
        target_mw = (property_constraints['molecularWeight']['min'] + property_constraints['molecularWeight']['max']) / 2
        
        print(f"Target properties - LogP: {target_logp:.2f}, QED: {target_qed:.2f}, MW: {target_mw:.2f}")
        
        # Get molecule embedding
        embedding = self.get_molecule_embedding(smiles)
        
        # Initialize best molecule
        best_smiles = smiles
        best_mol = mol
        best_score = self._calculate_score(mol, property_constraints)
        
        # Collect data for SR retraining
        optimization_data = {
            'embeddings': [],
            'logp_values': [],
            'qed_values': [],
            'mw_values': [],
            'scores': []
        }
        
        # Add initial molecule to data
        optimization_data['embeddings'].append(embedding)
        optimization_data['logp_values'].append(initial_logp)
        optimization_data['qed_values'].append(initial_qed)
        optimization_data['mw_values'].append(initial_mw)
        optimization_data['scores'].append(best_score)
        
        # Optimization loop with SR guidance
        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}")
            
            # Every 5 iterations, retrain SR models with collected data
            if i > 0 and i % 5 == 0 and len(optimization_data['embeddings']) > 5:
                print("Retraining SR models with collected data")
                self._retrain_sr_models(optimization_data)
            
            # Generate variations in latent space
            perturbation_scale = 0.3 * (1.0 - i/iterations)
            perturbation = np.random.normal(0, perturbation_scale, embedding.shape)
            perturbed_embedding = embedding + perturbation
            
            # Use SR models to predict properties of perturbed embedding
            pred_properties = self.predict_properties_with_sr(perturbed_embedding)
            
            # Calculate SR-predicted score
            sr_logp_score = abs(pred_properties['logP'] - target_logp) / max(0.1, property_constraints['logP']['max'] - property_constraints['logP']['min'])
            sr_qed_score = abs(pred_properties['QED'] - target_qed) / max(0.1, property_constraints['QED']['max'] - property_constraints['QED']['min'])
            sr_mw_score = abs(pred_properties['MolecularWeight'] - target_mw) / max(10.0, property_constraints['molecularWeight']['max'] - property_constraints['molecularWeight']['min'])
            sr_score = 0.4 * sr_logp_score + 0.4 * sr_qed_score + 0.2 * sr_mw_score
            
            # Only continue with promising directions in latent space
            if sr_score > best_score * 1.5:  # If SR predicts the direction is much worse
                continue
            
            # Try to decode directly using the VAE first
            try:
                new_smiles = self.decode_from_latent(perturbed_embedding)
                if new_smiles and Chem.MolFromSmiles(new_smiles):
                    # Successfully decoded a valid molecule
                    new_mol = Chem.MolFromSmiles(new_smiles)
                else:
                    # If direct decoding failed, find similar molecule in reference set
                    new_smiles = self._find_similar_molecule(perturbed_embedding, property_constraints)
                    if not new_smiles:
                        continue
                    new_mol = Chem.MolFromSmiles(new_smiles)
                    if new_mol is None:
                        continue
            except:
                # Find similar molecule in reference set as fallback
                new_smiles = self._find_similar_molecule(perturbed_embedding, property_constraints)
                if not new_smiles:
                    continue
                new_mol = Chem.MolFromSmiles(new_smiles)
                if new_mol is None:
                    continue
                
            # Calculate actual properties
            logp = Descriptors.MolLogP(new_mol)
            qed = QED.qed(new_mol)
            mw = Descriptors.MolWt(new_mol)
                
            # Calculate actual score
            score = self._calculate_score(new_mol, property_constraints)
            
            # Add to optimization data for SR training
            new_embedding = self.get_molecule_embedding(new_smiles)
            optimization_data['embeddings'].append(new_embedding)
            optimization_data['logp_values'].append(logp)
            optimization_data['qed_values'].append(qed)
            optimization_data['mw_values'].append(mw)
            optimization_data['scores'].append(score)
            
            # Check if better than current best
            if score < best_score:
                best_score = score
                best_smiles = new_smiles
                best_mol = new_mol
                # Update the embedding to focus search around this better molecule
                embedding = new_embedding
                print(f"Found better molecule with score {score:.4f}")
                print(f"Properties - LogP: {logp:.2f}, QED: {qed:.2f}, MW: {mw:.2f}")
        
        # Final properties
        final_logp = Descriptors.MolLogP(best_mol)
        final_qed = QED.qed(best_mol)
        final_mw = Descriptors.MolWt(best_mol)
        
        print(f"Refinement complete")
        print(f"Final properties - LogP: {final_logp:.2f}, QED: {final_qed:.2f}, MW: {final_mw:.2f}")
        
        # Ensure canonical SMILES
        best_smiles = Chem.MolToSmiles(best_mol, isomericSmiles=True)
        return best_smiles

    def _retrain_sr_models(self, optimization_data):
        """
        Retrain SR models with collected optimization data
        
        Args:
            optimization_data (dict): Dictionary containing collected data
        """
        try:
            # Convert data to arrays
            X = np.array(optimization_data['embeddings'])
            y_logp = np.array(optimization_data['logp_values'])
            y_qed = np.array(optimization_data['qed_values'])
            y_mw = np.array(optimization_data['mw_values'])
            
            # Use PCA to reduce dimensionality for SR
            if self.pca is None:
                self.pca = PCA(n_components=min(10, X.shape[0] - 1))
                X_reduced = self.pca.fit_transform(X)
            else:
                X_reduced = self.pca.transform(X)
            
            # Retrain LogP model with warm start if possible
            if self.sr_models['logP'] is None:
                self.sr_models['logP'] = SymbolicRegressor(
                    population_size=300, 
                    generations=10, 
                    parsimony_coefficient=0.01,
                    n_jobs=-1
                )
            self.sr_models['logP'].fit(X_reduced, y_logp)
            
            # Retrain QED model
            if self.sr_models['QED'] is None:
                self.sr_models['QED'] = SymbolicRegressor(
                    population_size=300, 
                    generations=10, 
                    parsimony_coefficient=0.01,
                    n_jobs=-1
                )
            self.sr_models['QED'].fit(X_reduced, y_qed)
            
            # Retrain MW model
            if self.sr_models['MolecularWeight'] is None:
                self.sr_models['MolecularWeight'] = SymbolicRegressor(
                    population_size=300, 
                    generations=10, 
                    parsimony_coefficient=0.01,
                    n_jobs=-1
                )
            self.sr_models['MolecularWeight'].fit(X_reduced, y_mw)
            
            # Print model scores to track improvement
            print(f"SR models retrained - LogP: {self.sr_models['logP'].score(X_reduced, y_logp):.4f}, "
                  f"QED: {self.sr_models['QED'].score(X_reduced, y_qed):.4f}, "
                  f"MW: {self.sr_models['MolecularWeight'].score(X_reduced, y_mw):.4f}")
                  
        except Exception as e:
            print(f"Error retraining SR models: {e}")
    
    def _calculate_score(self, mol, property_constraints):
        """
        Calculate score for a molecule based on distance from target properties
        
        Args:
            mol (rdkit.Chem.Mol): RDKit molecule
            property_constraints (dict): Dictionary with property constraints
            
        Returns:
            float: Score (lower is better)
        """
        try:
            # Calculate properties
            logp = Descriptors.MolLogP(mol)
            qed = QED.qed(mol)
            mw = Descriptors.MolWt(mol)
            
            # Get target property values (midpoint of ranges)
            target_logp = (property_constraints['logP']['min'] + property_constraints['logP']['max']) / 2
            target_qed = (property_constraints['QED']['min'] + property_constraints['QED']['max']) / 2
            target_mw = (property_constraints['molecularWeight']['min'] + property_constraints['molecularWeight']['max']) / 2
            
            # Calculate normalized distances from targets
            logp_score = abs(logp - target_logp) / max(0.1, property_constraints['logP']['max'] - property_constraints['logP']['min'])
            qed_score = abs(qed - target_qed) / max(0.1, property_constraints['QED']['max'] - property_constraints['QED']['min'])
            mw_score = abs(mw - target_mw) / max(10.0, property_constraints['molecularWeight']['max'] - property_constraints['molecularWeight']['min'])
            
            # Weighted score - give higher importance to drug-likeness (QED) and LogP
            score = 0.4 * logp_score + 0.4 * qed_score + 0.2 * mw_score
            
            return score
        except Exception as e:
            print(f"Error calculating score: {str(e)}")
            return float('inf')  # Return infinity for invalid molecules

    def _find_similar_molecule(self, embedding, property_constraints):
        """
        Find a similar molecule in the latent space that satisfies property constraints
        
        Args:
            embedding (numpy.ndarray): Embedding vector
            property_constraints (dict): Dictionary with property constraints
            
        Returns:
            str: SMILES string of similar molecule or None if not found
        """
        # Get target property values (midpoint of ranges)
        target_logp = (property_constraints['logP']['min'] + property_constraints['logP']['max']) / 2
        
        # Get reference molecules for logP property (most important constraint)
        ref_smiles, ref_embeddings = self.get_reference_embeddings('logP', target_logp)
        
        if len(ref_smiles) == 0:
            return None
        
        # Calculate similarity scores
        similarities = cosine_similarity([embedding], ref_embeddings)[0]
        
        # Sort by similarity
        sorted_indices = np.argsort(-similarities)  # Descending order
        
        # Try to find a molecule that satisfies all constraints
        for idx in sorted_indices:
            smiles = ref_smiles[idx]
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                    
                # Calculate properties
                logp = Descriptors.MolLogP(mol)
                qed = QED.qed(mol)
                mw = Descriptors.MolWt(mol)
                
                # Check constraints
                if (property_constraints['logP']['min'] <= logp <= property_constraints['logP']['max'] and
                    property_constraints['QED']['min'] <= qed <= property_constraints['QED']['max'] and
                    property_constraints['molecularWeight']['min'] <= mw <= property_constraints['molecularWeight']['max']):
                    return smiles
            except:
                continue
        
        # If no molecule satisfies all constraints, return the most similar one
        return ref_smiles[sorted_indices[0]]


# Standalone wrapper function for compatibility with the rest of the pipeline
def refine_molecule_with_vae(smiles, property_constraints, iterations=20):
    """
    Refine a molecule using the MoleculeGenerator class with SR-guided optimization.
    
    Args:
        smiles (str): Initial SMILES string
        property_constraints (dict): Dictionary with property constraints
        iterations (int): Number of optimization iterations
        
    Returns:
        str: Optimized SMILES string
    """
    # Create a generator instance and call its method
    generator = MoleculeGenerator()
    return generator.refine_molecule(smiles, property_constraints, iterations)


# Old implementation commented out for reference if needed
"""
def refine_molecule_with_vae_demo(smiles, property_constraints, iterations=20):
    \"""
    Demo version that simulates refinement using time delay.
    This function is for demonstration purposes only.
    \"""
    # ...existing code...
"""