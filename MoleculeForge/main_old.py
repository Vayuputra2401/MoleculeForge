import streamlit as st
import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, PandasTools
from google import genai
import json

# Set page config
st.set_page_config(page_title="AI-Powered Molecular Generation", layout="wide")

try:
    import py3Dmol
    from stmol import showmol
    ENABLE_3D = True
except ImportError:
    ENABLE_3D = False
    st.warning("3D visualization libraries are missing. Install them with: `pip install py3Dmol stmol ipython_genutils`")

# Import custom modules
from molecular_generator import generate_molecules_from_prompt
from vae_model import refine_molecule_with_vae
from hypergrammar_validator import validate_molecule
from property_calculator import calculate_properties
from visualization import visualize_molecule_3d

# Secrets management system
def get_secret(key):
    # Path to the secrets file
    secrets_path = os.path.join(os.path.dirname(__file__), 'secrets.json')
    
    # Create secrets file if it doesn't exist
    if not os.path.exists(secrets_path):
        with open(secrets_path, 'w') as f:
            json.dump({}, f)
    
    # Read secrets file
    try:
        with open(secrets_path, 'r') as f:
            secrets = json.load(f)
            return secrets.get(key)
    except:
        return None

def save_secret(key, value):
    # Path to the secrets file
    secrets_path = os.path.join(os.path.dirname(__file__), 'secrets.json')
    
    # Read existing secrets
    try:
        with open(secrets_path, 'r') as f:
            secrets = json.load(f)
    except:
        secrets = {}
    
    # Update secret
    secrets[key] = value
    
    # Write back to file
    with open(secrets_path, 'w') as f:
        json.dump(secrets, f)

# Configure Gemini API - use client approach
def configure_gemini(api_key=None):
    if (api_key):
        try:
            # Initialize the client with the provided API key
            client = genai.Client(api_key=api_key)
            # Save API key to secrets
            save_secret("gemini_api_key", api_key)
            return client
        except Exception as e:
            st.sidebar.error(f"Error configuring Gemini API: {str(e)}")
            return None
    else:
        st.warning("Please enter your Gemini API Key in the sidebar to use AI generation.")
        return None

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/5f/Siemens-logo.svg", width=200)
    st.markdown("## Molecular Generation Pipeline")
    st.markdown("---")
    
    # Check if API key already exists in secrets
    saved_api_key = get_secret("gemini_api_key")
    
    # Add API key input to the sidebar, pre-filled if available
    api_key = st.text_input(
        "Enter your Gemini API Key:",
        type="password",
        value=saved_api_key if saved_api_key else ""
    )
    
    # User inputs
    st.markdown("### Target Parameters")
    target_description = st.text_area(
        "Describe the target molecule:", 
        value="A JAK inhibitor with high selectivity for JAK1, moderate lipophilicity, and good oral bioavailability.",
        height=150
    )
    
    # Add specific property constraints
    st.markdown("### Property Constraints")
    col1, col2 = st.columns(2)
    with col1:
        logp_min = st.number_input("LogP (min)", value=1.0, step=0.1)
        mol_weight_min = st.number_input("Molecular Weight (min)", value=200.0, step=10.0)
        qed_min = st.number_input("QED (min)", value=0.5, step=0.05)
    
    with col2:
        logp_max = st.number_input("LogP (max)", value=4.0, step=0.1)
        mol_weight_max = st.number_input("Molecular Weight (max)", value=500.0, step=10.0)
        qed_max = st.number_input("QED (max)", value=0.9, step=0.05)
    
    advanced_options = st.expander("Advanced Options")
    with advanced_options:
        num_molecules = st.slider("Number of molecules to generate", 1, 10, 3)
        vae_iterations = st.slider("VAE refinement iterations", 5, 50, 20)
    
    # Generate button
    generate_button = st.button("Generate Molecules", type="primary")

# Main content
st.title("AI-Powered Molecular Generation Pipeline")

# Initialize session state
if 'generated_molecules' not in st.session_state:
    st.session_state.generated_molecules = []
    st.session_state.validated_molecules = []
    st.session_state.molecule_properties = []
    st.session_state.current_molecule_index = 0

# Function to reset state
def reset_state():
    st.session_state.generated_molecules = []
    st.session_state.validated_molecules = []
    st.session_state.molecule_properties = []
    st.session_state.current_molecule_index = 0

# Main pipeline execution
if generate_button:
    try:
        reset_state()
        
        # Configure Gemini client
        genai_client = configure_gemini(api_key)
        
        # Display pipeline steps
        pipeline_progress = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Generate initial SMILES from NLU
        status_text.text("Step 1/4: Generating initial molecules from description...")
        
        # Prepare property constraints
        property_constraints = {
            "logP": {"min": logp_min, "max": logp_max},
            "molecularWeight": {"min": mol_weight_min, "max": mol_weight_max},
            "QED": {"min": qed_min, "max": qed_max}
        }
        
        # Generate molecules with the new client
        generated_smiles = generate_molecules_from_prompt(
            target_description, 
            property_constraints,
            genai_client, 
            num_molecules=num_molecules
        )
        st.write(generated_smiles)
        pipeline_progress.progress(25)
        
        # Step 2: Refine molecules with VAE
        status_text.text("Step 2/4: Refining molecules with VAE guided by Symbolic Regression...")
        
        refined_molecules = []
        for smiles in generated_smiles:
            refined_smiles = refine_molecule_with_vae(
                smiles, 
                property_constraints, 
                iterations=vae_iterations
            )
            refined_molecules.append(refined_smiles)
        
        pipeline_progress.progress(50)
        
        # Step 3: Validate with hypergrammar
        status_text.text("Step 3/4: Validating molecules with hypergrammar...")
        
        validated_molecules = []
        for smiles in refined_molecules:
            # Add debug information
            if smiles is None:
                st.warning(f"Found a None value in refined molecules. This shouldn't happen.")
                is_valid, validation_message = False, "Invalid: None value received"
            else:
                try:
                    is_valid, validation_message = validate_molecule(smiles)
                except Exception as e:
                    st.warning(f"Error validating molecule: {str(e)}")
                    is_valid, validation_message = False, f"Error during validation: {str(e)}"
            
            validated_molecules.append({
                "smiles": smiles, 
                "validation": "Valid" if is_valid else "Invalid", 
                "message": validation_message
            })
        
        pipeline_progress.progress(75)
        
        # Step 4: Calculate properties
        status_text.text("Step 4/4: Calculating molecular properties...")
        
        molecule_properties = []
        for mol_data in validated_molecules:
            if mol_data["validation"] == "Valid":
                properties = calculate_properties(mol_data["smiles"])
                molecule_properties.append(properties)
            else:
                molecule_properties.append(None)
        
        pipeline_progress.progress(100)
        status_text.text("Pipeline completed successfully!")
        
        # Store results in session state
        st.session_state.generated_molecules = refined_molecules
        st.session_state.validated_molecules = validated_molecules
        st.session_state.molecule_properties = molecule_properties
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

# Display results
if st.session_state.generated_molecules:
    st.markdown("## Generated Molecules")
    
    # Create tabs for each molecule
    molecule_tabs = st.tabs([f"Molecule {i+1}" for i in range(len(st.session_state.generated_molecules))])
    
    for i, tab in enumerate(molecule_tabs):
        with tab:
            mol_smiles = st.session_state.generated_molecules[i]
            validation_data = st.session_state.validated_molecules[i]
            properties = st.session_state.molecule_properties[i]
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Molecule Structure")
                
                # Display SMILES
                st.markdown(f"**SMILES**: `{mol_smiles}`")
                
                # Display 2D structure
                mol = Chem.MolFromSmiles(mol_smiles)
                if mol:
                    img = Draw.MolToImage(mol, size=(400, 300))
                    st.image(img, caption="2D Structure")
                else:
                    st.warning("Could not generate 2D structure")
                
                # Display validation status
                if validation_data["validation"] == "Valid":
                    st.success(f"Validation: {validation_data['validation']}")
                    st.info(validation_data["message"])
                else:
                    st.error(f"Validation: {validation_data['validation']}")
                    st.warning(validation_data["message"])
            
            with col2:
                st.markdown("### 3D Visualization")
                if validation_data["validation"] == "Valid" and mol:
                    # Generate 3D coordinates
                    try:
                        mol_3d = Chem.AddHs(mol)
                        AllChem.EmbedMolecule(mol_3d, randomSeed=42)
                        AllChem.MMFFOptimizeMolecule(mol_3d)
                        
                        # Create py3Dmol view
                        viewer = py3Dmol.view(width=400, height=400)
                        mol_block = Chem.MolToMolBlock(mol_3d)
                        viewer.addModel(mol_block, "mol")
                        viewer.setStyle({'stick': {}})
                        viewer.setStyle({'atom': 'C'}, {'color': 'gray'})
                        viewer.setStyle({'atom': 'O'}, {'color': 'red'})
                        viewer.setStyle({'atom': 'N'}, {'color': 'blue'})
                        viewer.setStyle({'atom': 'S'}, {'color': 'yellow'})
                        viewer.setStyle({'atom': 'Cl'}, {'color': 'green'})
                        viewer.setStyle({'atom': 'F'}, {'color': 'green'})
                        viewer.setStyle({'atom': 'Br'}, {'color': 'brown'})
                        viewer.setStyle({'atom': 'I'}, {'color': 'purple'})
                        viewer.zoomTo()
                        showmol(viewer, height=400, width=400)
                    except Exception as e:
                        st.warning(f"Could not generate 3D visualization: {str(e)}")
                else:
                    st.warning("3D visualization not available for invalid molecules")
            
            # Display properties
            if properties:
                st.markdown("### Molecular Properties")
                
                # Create property table
                prop_df = pd.DataFrame([properties])
                
                # Format the table
                formatted_df = prop_df.round(2)
                
                # Display as a table
                st.table(formatted_df)
                
                # Create radar chart for key properties
                st.markdown("### Property Profile")
                
                # Select key properties for radar chart
                radar_props = {
                    'LogP': properties['logP'],
                    'QED': properties['QED'],
                    'MW': properties['MolecularWeight'] / 500,  # Normalize
                    'TPSA': properties['TPSA'] / 140,  # Normalize
                    'HBD': properties['NumHBondDonors'] / 5,  # Normalize
                    'HBA': properties['NumHBondAcceptors'] / 10,  # Normalize
                }
                
                # Create radar chart
                fig = plt.figure(figsize=(6, 6))
                ax = fig.add_subplot(111, polar=True)
                
                # Set the angles of the plot
                angles = np.linspace(0, 2*np.pi, len(radar_props), endpoint=False).tolist()
                values = list(radar_props.values())
                values += values[:1]  # Close the plot
                angles += angles[:1]  # Close the plot
                
                # Plot data
                ax.plot(angles, values, 'o-', linewidth=2)
                ax.fill(angles, values, alpha=0.25)
                
                # Set labels
                ax.set_thetagrids(np.degrees(angles[:-1]), list(radar_props.keys()))
                
                # Set y-axis limits
                ax.set_ylim(0, 1.2)
                
                st.pyplot(fig)
            else:
                st.warning("Property calculation not available for invalid molecules")

# Download section
if st.session_state.generated_molecules:
    st.markdown("## Export Results")
    
    # Create a dataframe of all molecules and their properties
    export_data = []
    for i, smiles in enumerate(st.session_state.generated_molecules):
        validation = st.session_state.validated_molecules[i]
        properties = st.session_state.molecule_properties[i]
        
        data_row = {
            "SMILES": smiles,
            "Validation": validation["validation"],
            "ValidationMessage": validation["message"]
        }
        
        if properties:
            data_row.update(properties)
        
        export_data.append(data_row)
    
    export_df = pd.DataFrame(export_data)
    
    # Download button
    csv = export_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="generated_molecules.csv",
        mime="text/csv"
    )

# Add footer
st.markdown("---")
st.markdown("### About This Tool")
st.markdown("""
This prototype demonstrates an AI-powered molecular generation pipeline that combines:
1. Natural language understanding for target specification
2. VAE-based molecule generation and refinement
3. Hypergrammar validation to ensure chemical feasibility
4. Property calculation and visualization
""")