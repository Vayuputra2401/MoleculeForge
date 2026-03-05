import streamlit as st
import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, PandasTools, Lipinski, QED
from google import genai
import json
import warnings

# Suppress warnings related to ChemBERTa decoding
warnings.filterwarnings("ignore", message=".*object has no attribute 'prefix'.*")

# Set page config with modern styling
st.set_page_config(
    page_title="AI-Powered Molecular Generation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .property-tooltip {
        font-size: 16px;
        color: #ffffff;
        background-color: #333333;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .property-card {
        background-color: #1a1a1a;
        border-radius: 5px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffffff;
        margin: 20px 0 10px 0;
    }
    .info-box {
        background-color: #333333;
        border-left: 5px solid #0096FF;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 3px;
    }
    .grid-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
    }
    .property-value {
        font-size: 24px;
        font-weight: bold;
        color: #ffffff;
    }
    .property-label {
        font-size: 14px;
        color: #cccccc;
    }
    .tooltip-icon {
        color: #9ca3af;
        font-size: 16px;
        margin-left: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Check for 3D visualization support
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

# Helper function to create tooltips for properties
def create_property_tooltip(property_name):
    tooltips = {
        "logP": "Partition coefficient (lipophilicity measure). Values between 2-5 are typically optimal for oral drugs (Lipinski's rule: <5).",
        "QED": "Quantitative Estimate of Drug-likeness. Ranges from 0 (not drug-like) to 1 (very drug-like).",
        "MolecularWeight": "The molecular mass in Daltons. Drug-like molecules typically have MW < 500 Da (Lipinski's rule).",
        "TPSA": "Topological Polar Surface Area in Å². Relates to cell membrane permeability. Values < 140 Å² for oral drugs.",
        "HBondDonors": "Number of hydrogen bond donors. Should be ≤ 5 for good oral bioavailability (Lipinski's rule).",
        "HBondAcceptors": "Number of hydrogen bond acceptors. Should be ≤ 10 for good oral bioavailability (Lipinski's rule).",
        "RotatableBonds": "Number of rotatable bonds. Related to molecular flexibility and oral bioavailability.",
        "AromaticRings": "Number of aromatic rings. Many drugs contain 1-3 aromatic rings.",
        "HeavyAtoms": "Count of non-hydrogen atoms. Drug-like molecules typically have 20-70 heavy atoms.",
        "FractionCSP3": "Fraction of carbon atoms that are sp3 hybridized. Higher values indicate more 3D character."
    }
    return tooltips.get(property_name, "No description available")

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
    
    # Help section collapsible
    with st.expander("ℹ️ How to use this tool"):
        st.markdown("""
        1. **Enter your Google Gemini API Key** (required for AI generation)
        2. **Describe your target molecule** using natural language
        3. **Adjust property constraints** to guide the generation
        4. **Click 'Generate Molecules'** to start the pipeline
        5. **Explore results** in the tabs that appear below
        
        This tool combines AI language models, VAE-based molecular optimization, 
        and chemoinformatics to generate drug-like molecules from text descriptions.
        """)
    
    # Check if API key already exists in secrets
    saved_api_key = get_secret("gemini_api_key")
    
    # Add API key input to the sidebar, pre-filled if available
    api_key = st.text_input(
        "Enter your Gemini API Key:",
        type="password",
        value=saved_api_key if saved_api_key else "",
        help="Required for molecule generation. Get your key from https://aistudio.google.com/"
    )
    
    # User inputs
    st.markdown("### Target Parameters")
    target_description = st.text_area(
        "Describe the target molecule:", 
        value="A JAK inhibitor with high selectivity for JAK1, moderate lipophilicity, and good oral bioavailability.",
        height=150,
        help="Use natural language to describe the desired molecule. Include target activity, selectivity, and desired properties."
    )
    
    # Add specific property constraints with explanations
    st.markdown("### Property Constraints")
    
    # Help text for property constraints
    st.info("These constraints guide the optimization process. The generated molecules will try to meet these criteria.")
    
    col1, col2 = st.columns(2)
    with col1:
        logp_min = st.number_input("LogP (min)", value=1.0, step=0.1, help="Minimum lipophilicity value (water-octanol partition coefficient)")
        mol_weight_min = st.number_input("Molecular Weight (min)", value=200.0, step=10.0, help="Minimum molecular weight in Daltons")
        qed_min = st.number_input("QED (min)", value=0.5, step=0.05, help="Minimum drug-likeness score (0-1)")
    
    with col2:
        logp_max = st.number_input("LogP (max)", value=4.0, step=0.1, help="Maximum lipophilicity value")
        mol_weight_max = st.number_input("Molecular Weight (max)", value=500.0, step=10.0, help="Maximum molecular weight (Lipinski: <500)")
        qed_max = st.number_input("QED (max)", value=0.9, step=0.05, help="Maximum drug-likeness score")
    
    advanced_options = st.expander("Advanced Options")
    with advanced_options:
        num_molecules = st.slider("Number of molecules to generate", 1, 10, 3, help="More molecules provide variety but take longer to generate")
        vae_iterations = st.slider("VAE refinement iterations", 5, 50, 20, help="More iterations can improve results but take longer")
    
    # Generate button
    generate_button = st.button("Generate Molecules", type="primary")

# Main content
st.markdown('<h1 class="main-header">AI-Powered Molecular Generation Pipeline</h1>', unsafe_allow_html=True)

# Short explanation of the pipeline
st.markdown("""
<div class="info-box">
<strong>How it works:</strong> This tool uses AI to transform your text description into potential drug molecules
through a four-stage pipeline:
<ol>
    <li><strong>AI Generation:</strong> Your description is analyzed by Google's Gemini AI to propose initial molecular structures</li>
    <li><strong>VAE Refinement:</strong> A Variational Autoencoder optimizes the molecules to meet your property constraints</li>
    <li><strong>Chemical Validation:</strong> Structures are validated against chemical grammar rules for synthetic feasibility</li>
    <li><strong>Property Analysis:</strong> Chemical properties are calculated and visualized for each candidate</li>
</ol>
</div>
""", unsafe_allow_html=True)

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
        
        if not genai_client:
            st.error("Gemini API configuration failed. Please check your API key.")
            st.stop()
        
        # Display pipeline steps with progress tracking
        pipeline_col1, pipeline_col2 = st.columns([3, 1])
        
        with pipeline_col1:
            pipeline_progress = st.progress(0)
            status_text = st.empty()
        
        with pipeline_col2:
            stage_indicator = st.empty()
            stage_indicator.markdown("⏳ **Stage 1/4**: Initial Generation")
        
        # Step 1: Generate initial SMILES from NLU
        status_text.text("Step 1/4: Generating initial molecules from description...")
        
        # Prepare property constraints
        property_constraints = {
            "logP": {"min": logp_min, "max": logp_max},
            "molecularWeight": {"min": mol_weight_min, "max": mol_weight_max},
            "QED": {"min": qed_min, "max": qed_max}
        }
        
        # Generate molecules with the new client
        generated_smiles , response = generate_molecules_from_prompt(
            target_description, 
            property_constraints,
            genai_client, 
            num_molecules=num_molecules
        )
        st.session_state.generated_molecules = generated_smiles
        # Display the raw response in a collapsible section for debugging
        with st.expander("Show AI Generation Response"):
            st.code(response, language="json")
        
        # Display the generated SMILES in a more visually appealing way
        st.subheader("Generated SMILES Structures")
        
        # Create a formatted table to display the SMILES
        smiles_df = pd.DataFrame({"SMILES": generated_smiles})
        st.dataframe(
            smiles_df,
            use_container_width=True,
            column_config={
            "SMILES": st.column_config.TextColumn(
                "Generated SMILES String",
                help="Simplified Molecular Input Line Entry System representation"
            )
            }
        )
        
        # Add a visual indicator of progress
        st.success(f"✅ Successfully generated {len(generated_smiles)} initial molecular structures")
        pipeline_progress.progress(25)
        stage_indicator.markdown("⏳ **Stage 2/4**: VAE Refinement")
        
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
        stage_indicator.markdown("⏳ **Stage 3/4**: Chemical Validation")
        
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
        stage_indicator.markdown("⏳ **Stage 4/4**: Property Analysis")
        
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
        stage_indicator.markdown("✅ **Pipeline Complete!**")
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
    st.markdown('<h2 class="section-header">Generated Molecules</h2>', unsafe_allow_html=True)
    
    # Create tabs for each molecule
    molecule_tabs = st.tabs([f"Molecule {i+1}" for i in range(len(st.session_state.generated_molecules))])
    
    for i, tab in enumerate(molecule_tabs):
        with tab:
            mol_smiles = st.session_state.generated_molecules[i]
            validation_data = st.session_state.validated_molecules[i]
            properties = st.session_state.molecule_properties[i]
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown('<h3 class="section-header">Molecule Structure</h3>', unsafe_allow_html=True)
                
                # Display SMILES with copy button
                st.code(mol_smiles, language="text")
                st.caption("SMILES (Simplified Molecular Input Line Entry System) is a notation that represents molecular structures as text strings.")
                
                # Display 2D structure
                mol = Chem.MolFromSmiles(mol_smiles)
                if mol:
                    img = Draw.MolToImage(mol, size=(400, 300), kekulize=True, wedgeBonds=True)
                    st.image(img, caption="2D Structure")
                    
                    # Add molecular formula
                    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
                    st.markdown(f"**Molecular Formula:** {formula}")
                else:
                    st.warning("Could not generate 2D structure")
                
                # Display validation status with improved styling
                if validation_data["validation"] == "Valid":
                    st.success(f"✅ Validation: {validation_data['validation']}")
                    with st.expander("What makes this molecule valid?"):
                        st.info(validation_data["message"])
                        st.markdown("""
                        A valid molecule:
                        - Has chemically feasible bonds and structures
                        - Contains only stable atomic configurations
                        - Follows standard valence rules
                        - Has reasonable ring structures and strain energies
                        """)
                else:
                    st.error(f"❌ Validation: {validation_data['validation']}")
                    with st.expander("Why is this molecule invalid?"):
                        st.warning(validation_data["message"])
                        st.markdown("""
                        Invalid molecules may contain:
                        - Impossible bond configurations
                        - Unstable ring structures
                        - Incorrect valence states
                        - Chemically unfeasible groups
                        """)
            
            with col2:
                st.markdown('<h3 class="section-header">3D Structure</h3>', unsafe_allow_html=True)
                
                # Replace the 3D visualization code in the main2.py file

                if validation_data["validation"] == "Valid" and mol:
                    if ENABLE_3D:
                        # Create a placeholder for the 3D view
                        viz_placeholder = st.empty()
                        
                        # Show a progress indicator
                        with st.spinner("Generating 3D model (this may take a few seconds)..."):
                            try:
                                # Generate 3D coordinates directly
                                mol_3d = Chem.AddHs(mol)
                                
                                # Try multiple embedding methods
                                embed_result = AllChem.EmbedMolecule(mol_3d, randomSeed=42)
                                if embed_result == -1:
                                    embed_result = AllChem.EmbedMolecule(mol_3d, useRandomCoords=True)
                                    if embed_result == -1:
                                        # Last resort: use ETKDG method which is more reliable
                                        AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
                                
                                # Optimize the structure
                                try:
                                    AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=200)
                                except:
                                    # Try UFF if MMFF fails
                                    try:
                                        AllChem.UFFOptimizeMolecule(mol_3d, maxIters=200)
                                    except:
                                        pass
                                
                                # Convert to mol block - the string representation for 3D molecules
                                mol_block = Chem.MolToMolBlock(mol_3d)
                                
                                # Create a py3Dmol view with explicit settings for better visibility
                                view = py3Dmol.view(width=400, height=400)
                                view.addModel(mol_block, "mol")
                                
                                # Use more visible styles with contrasting colors
                                view.setStyle({'stick': {'radius': 0.2, 'color': 'grey'}})
                                view.setStyle({'atom': 'C'}, {'sphere': {'scale': 0.3, 'color': 'grey'}})
                                view.setStyle({'atom': 'O'}, {'sphere': {'scale': 0.4, 'color': 'red'}})
                                view.setStyle({'atom': 'N'}, {'sphere': {'scale': 0.4, 'color': 'blue'}})
                                view.setStyle({'atom': 'S'}, {'sphere': {'scale': 0.5, 'color': 'yellow'}})
                                view.setStyle({'atom': 'Cl'}, {'sphere': {'scale': 0.5, 'color': 'green'}})
                                view.setStyle({'atom': 'F'}, {'sphere': {'scale': 0.3, 'color': 'lime'}})
                                view.setStyle({'atom': 'Br'}, {'sphere': {'scale': 0.5, 'color': 'brown'}})
                                
                                # Add other visualization enhancements
                                view.setBackgroundColor('white')  # Use black background for better contrast
                                view.zoomTo()  # Auto-zoom to fit the molecule
                                
                                # Generate the HTML content
                                view_html = view.write_html()
                                
                                # Use the stmol package which is specifically designed for py3Dmol in Streamlit
                                from stmol import showmol
                                showmol(view, height=450, width=450)
                                
                                # Also offer a fallback display option
                                with st.expander("Can't see the 3D view? Try this alternative"):
                                    st.download_button(
                                        "Download 3D Model HTML",
                                        data=view_html,
                                        file_name=f"molecule_{i+1}_3d.html",
                                        mime="text/html"
                                    )
                                    st.info("Download and open the HTML file in your browser to view the 3D model")
                            
                            except Exception as e:
                                # Handle any other errors
                                st.error(f"Error generating 3D visualization: {str(e)}")
                                import traceback
                                st.error(traceback.format_exc())
                                st.image(Draw.MolToImage(mol, size=(400, 300)), caption="2D Structure (Fallback)")
                    else:
                        # 3D visualization not enabled
                        st.warning("3D visualization requires additional packages.")
                        st.info("Install required packages with:\n`pip install py3Dmol stmol ipython_genutils`")
                        st.image(Draw.MolToImage(mol, size=(400, 300)), caption="2D Structure (3D view not available)")
                else:
                    # Invalid molecule
                    st.warning("3D visualization not available for invalid molecules")
            
            # Display properties with improved UI
            if properties:
                st.markdown('<h3 class="section-header">Molecular Properties</h3>', unsafe_allow_html=True)
                
                # Display key property cards in a grid
                st.markdown('<div class="property-card">', unsafe_allow_html=True)
                cols = st.columns(3)
                
                # Key properties with explanations
                with cols[0]:
                    st.metric("LogP", f"{properties['logP']:.2f}")
                    st.markdown('<div class="property-tooltip">Lipophilicity measure. Optimal range: 1-5</div>', unsafe_allow_html=True)
                
                with cols[1]:
                    st.metric("QED", f"{properties['QED']:.2f}")
                    st.markdown('<div class="property-tooltip">Drug-likeness score. Higher is better (0-1)</div>', unsafe_allow_html=True)
                
                with cols[2]:
                    st.metric("MW", f"{properties['MolecularWeight']:.1f}")
                    st.markdown('<div class="property-tooltip">Molecular weight. Ideal: <500 Da</div>', unsafe_allow_html=True)
                
                # Second row of properties
                cols = st.columns(3)
                with cols[0]:
                    st.metric("TPSA", f"{properties['TPSA']:.1f}")
                    st.markdown('<div class="property-tooltip">Polar surface area. Good oral absorption: <140 Å²</div>', unsafe_allow_html=True)
                
                with cols[1]:
                    hbd = properties.get('HBondDonors', properties.get('NumHBondDonors', 0))
                    hba = properties.get('HBondAcceptors', properties.get('NumHBondAcceptors', 0))
                    st.metric("HB Donors/Acceptors", f"{hbd}/{hba}")
                    st.markdown('<div class="property-tooltip">Hydrogen bond donors/acceptors. Lipinski: ≤5/≤10</div>', unsafe_allow_html=True)
                
                with cols[2]:
                    rotb = properties.get('RotatableBonds', 0)
                    st.metric("Rotatable Bonds", f"{rotb}")
                    st.markdown('<div class="property-tooltip">Flexibility measure. Good oral bioavailability: ≤10</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Detailed property table with explanations - fixed formatting
                with st.expander("View All Properties"):
                    # Create a more readable property table with improved formatting
                    formatted_props = {}
                    for key, value in properties.items():
                        if isinstance(value, float):
                            formatted_props[key] = round(value, 3)
                        else:
                            formatted_props[key] = value
                    
                    prop_df = pd.DataFrame([formatted_props])
                    
                    # Use column_config to better format the table
                    st.dataframe(
                        prop_df,
                        use_container_width=True,
                        column_config={
                            col: st.column_config.NumberColumn(
                                col,
                                help=create_property_tooltip(col),
                                format="%.2f" if col not in ["RotatableBonds", "AromaticRings", "HBondDonors", "HBondAcceptors", "NumHBondDonors", "NumHBondAcceptors"] else "%d"
                            )
                            for col in prop_df.columns
                        }
                    )
                    
                    st.markdown("""
                    **Property Key:**
                    - **LogP**: Octanol-water partition coefficient (lipophilicity)
                    - **QED**: Quantitative Estimate of Drug-likeness (0-1)
                    - **MolecularWeight**: Molecular mass in Daltons
                    - **TPSA**: Topological Polar Surface Area in Å²
                    - **HBondDonors/NumHBondDonors**: Number of hydrogen bond donors
                    - **HBondAcceptors/NumHBondAcceptors**: Number of hydrogen bond acceptors
                    - **RotatableBonds**: Number of rotatable bonds
                    - **AromaticRings**: Number of aromatic rings
                    - **HeavyAtoms**: Number of non-hydrogen atoms
                    - **FractionCSP3**: Fraction of sp3 hybridized carbon atoms (complexity)
                    """)
                
                # Interactive Visualization Section
                st.markdown('<h3 class="section-header">Property Analysis</h3>', unsafe_allow_html=True)
                
                viz_tabs = st.tabs(["Radar Chart", "Drug-likeness", "Property Distribution"])
                
                with viz_tabs[0]:
                    # Radar chart with explanations
                    st.markdown("#### Molecular Property Profile")
                    st.markdown("This radar chart shows key molecular properties normalized to the 0-1 range, where values closer to the outer edge indicate better drug-like properties.")
                    
                    # Select key properties for radar chart with better error handling
                    radar_props = {}
                    
                    # Add properties that definitely exist
                    radar_props['LogP'] = properties.get('logP', 0)
                    radar_props['QED'] = properties.get('QED', 0)
                    radar_props['MW'] = properties.get('MolecularWeight', 0) / 500  # Normalize
                    radar_props['TPSA'] = properties.get('TPSA', 0) / 140  # Normalize
                    
                    # Try different naming conventions for H-bond donors and acceptors
                    for key in ['HBondDonors', 'NumHBondDonors']:
                        if key in properties:
                            radar_props['HBD'] = properties[key] / 5  # Normalize
                            break
                    
                    for key in ['HBondAcceptors', 'NumHBondAcceptors']:
                        if key in properties:
                            radar_props['HBA'] = properties[key] / 10  # Normalize
                            break
                    
                    # Create radar chart
                    fig = go.Figure()
                    
                    # Add radar chart trace
                    categories = list(radar_props.keys())
                    values = list(radar_props.values())
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name='Properties',
                        line=dict(color='rgb(31, 119, 180)', width=2),
                        fillcolor='rgba(31, 119, 180, 0.3)'
                    ))
                    
                    # Add reference trace for optimal values
                    optimal_values = {
                        'LogP': 0.6,  # Normalized optimal LogP (~3)
                        'QED': 0.8,   # High QED
                        'MW': 0.7,    # Normalized MW (~350)
                        'TPSA': 0.5,  # Normalized TPSA (~70)
                        'HBD': 0.4,   # Normalized HBD (~2)
                        'HBA': 0.5    # Normalized HBA (~5)
                    }
                    
                    # Ensure we have the same categories in the same order
                    optimal_r = [optimal_values.get(cat, 0) for cat in categories]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=optimal_r,
                        theta=categories,
                        name='Optimal Range',
                        line=dict(color='rgba(255, 99, 132, 0.8)', width=2, dash='dash'),
                        fillcolor='rgba(255, 99, 132, 0.1)',
                        fill='toself'
                    ))
                    
                    # Configure layout
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1.2]
                            ),
                        ),
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5),
                        title="Property Radar Chart (Normalized Values)"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("How to interpret this chart?"):
                        st.markdown("""
                        This radar chart shows key molecular properties normalized to a 0-1 scale:
                        - **LogP**: Value / optimal range (~2)
                        - **QED**: Already on 0-1 scale (higher is more drug-like)
                        - **MW**: Value / 500 (typical upper limit for small molecules)
                        - **TPSA**: Value / 140 (typical upper limit for oral bioavailability)
                        - **HBD**: Number of H-bond donors / 5 (Lipinski limit)
                        - **HBA**: Number of H-bond acceptors / 10 (Lipinski limit)
                        
                        The blue area shows this molecule's properties, while the red dashed line indicates optimal drug-like ranges. 
                        Ideal candidates have blue areas that closely match the red reference line.
                        """)
                
                with viz_tabs[1]:
                    # Drug-likeness analysis with Lipinski's Rule of 5
                    st.markdown("#### Drug-likeness Assessment")
                    st.markdown("This analysis evaluates how well the molecule meets established criteria for drug-likeness.")
                    
                    # Calculate Lipinski violations
                    lipinski_rules = {
                        "Molecular Weight < 500": properties['MolecularWeight'] < 500,
                        "LogP < 5": properties['logP'] < 5,
                        "H-Bond Donors ≤ 5": properties.get('NumHBondDonors', properties.get('HBondDonors', 0)) <= 5,
                        "H-Bond Acceptors ≤ 10": properties.get('NumHBondAcceptors', properties.get('HBondAcceptors', 0)) <= 10
                    }
                    
                    # Create bar chart for Lipinski compliance
                    fig = go.Figure()
                    
                    for rule, complies in lipinski_rules.items():
                        fig.add_trace(go.Bar(
                            x=[rule],
                            y=[1],
                            name=rule,
                            marker_color='green' if complies else 'red',
                            text="Pass" if complies else "Fail",
                            textposition="inside",
                            hoverinfo="text",
                            hovertext=f"{rule}: {'Pass' if complies else 'Fail'}"
                        ))
                    
                    # Configure layout
                    fig.update_layout(
                        title="Lipinski's Rule of 5 Compliance",
                        yaxis=dict(
                            showticklabels=False,
                            title="",
                            range=[0, 1]
                        ),
                        xaxis=dict(title=""),
                        showlegend=False,
                        barmode='group',
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key=f"lipinski_chart_{i}")
                    
                    # Count passed rules and show summary
                    rules_passed = sum(1 for rule in lipinski_rules.values() if rule)
                    
                    if rules_passed == 4:
                        st.success(f"✅ Passes all {rules_passed}/4 Lipinski rules - Excellent drug-likeness")
                    elif rules_passed == 3:
                        st.success(f"✅ Passes {rules_passed}/4 Lipinski rules - Good drug-likeness")
                    elif rules_passed == 2:
                        st.warning(f"⚠️ Passes only {rules_passed}/4 Lipinski rules - Fair drug-likeness")
                    else:
                        st.error(f"❌ Passes only {rules_passed}/4 Lipinski rules - Poor drug-likeness")
                    
                    with st.expander("About Lipinski's Rule of 5"):
                        st.markdown("""
                        Lipinski's Rule of 5 predicts that poor absorption or permeation is more likely when:
                        - Molecular weight > 500 Da
                        - LogP > 5 (too lipophilic)
                        - More than 5 hydrogen bond donors (sum of OH and NH groups)
                        - More than 10 hydrogen bond acceptors (sum of O and N atoms)
                        
                        Molecules that satisfy 3 or more of these rules are more likely to be orally bioavailable.
                        This rule set is widely used as a filter for drug-like properties.
                        """)
                
                with viz_tabs[2]:
                    # Add a new property distribution chart
                    st.markdown("#### Property Distribution")
                    
                    # Create bar chart showing property values compared to ideal ranges
                    property_data = {
                        "Property": ["LogP", "QED", "MW/100", "TPSA/10", "HB Donors", "HB Acceptors", "Rot. Bonds"],
                        "Value": [
                            properties['logP'],
                            properties['QED'],
                            properties['MolecularWeight']/100,  # Scaled for visualization
                            properties['TPSA']/10,  # Scaled for visualization
                            properties.get('HBondDonors', properties.get('NumHBondDonors', 0)),
                            properties.get('HBondAcceptors', properties.get('NumHBondAcceptors', 0)),
                            properties.get('RotatableBonds', 0)
                        ]
                    }
                    
                    # Define color thresholds for each property
                    def get_color(prop, val):
                        if prop == "LogP":
                            return 'green' if 2 <= val <= 5 else 'orange' if 1 <= val < 2 or 5 < val <= 6 else 'red'
                        elif prop == "QED":
                            return 'green' if val >= 0.7 else 'orange' if val >= 0.5 else 'red'
                        elif prop == "MW/100":
                            # Original MW is val*100
                            return 'green' if val*100 <= 500 else 'orange' if val*100 <= 550 else 'red'
                        elif prop == "TPSA/10":
                            # Original TPSA is val*10
                            return 'green' if val*10 <= 140 else 'orange' if val*10 <= 180 else 'red'
                        elif prop == "HB Donors":
                            return 'green' if val <= 5 else 'orange' if val <= 7 else 'red'
                        elif prop == "HB Acceptors":
                            return 'green' if val <= 10 else 'orange' if val <= 12 else 'red'
                        elif prop == "Rot. Bonds":
                            return 'green' if val <= 10 else 'orange' if val <= 12 else 'red'
                        return 'blue'
                    
                    # Create colors list
                    colors = [get_color(prop, val) for prop, val in zip(property_data["Property"], property_data["Value"])]
                    
                    # Create horizontal bar chart
                    fig = go.Figure(go.Bar(
                        x=property_data["Value"],
                        y=property_data["Property"],
                        orientation='h',
                        marker_color=colors,
                        text=[f"{x:.2f}" if isinstance(x, float) else str(x) for x in property_data["Value"]],
                        textposition='auto'
                    ))
                    
                    # Add range indicators for ideal values
                    annotations = [
                        dict(x=2, y="LogP", xref="x", yref="y", text="Min", showarrow=True, arrowhead=2, ax=-20, ay=0),
                        dict(x=5, y="LogP", xref="x", yref="y", text="Max", showarrow=True, arrowhead=2, ax=20, ay=0),
                        dict(x=0.7, y="QED", xref="x", yref="y", text="Good", showarrow=True, arrowhead=2, ax=20, ay=0),
                        dict(x=5, y="MW/100", xref="x", yref="y", text="Max 500", showarrow=True, arrowhead=2, ax=20, ay=0),
                        dict(x=14, y="TPSA/10", xref="x", yref="y", text="Max 140", showarrow=True, arrowhead=2, ax=20, ay=0),
                        dict(x=5, y="HB Donors", xref="x", yref="y", text="Max", showarrow=True, arrowhead=2, ax=20, ay=0),
                        dict(x=10, y="HB Acceptors", xref="x", yref="y", text="Max", showarrow=True, arrowhead=2, ax=20, ay=0),
                        dict(x=10, y="Rot. Bonds", xref="x", yref="y", text="Max", showarrow=True, arrowhead=2, ax=20, ay=0),
                    ]
                    
                    fig.update_layout(
                        title="Property Distribution vs Ideal Ranges",
                        xaxis_title="Value",
                        yaxis_title="Property",
                        height=400,
                        annotations=annotations
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("Understanding Property Colors"):
                        st.markdown("""
                        The bar colors indicate how well each property fits within ideal ranges for drug-like molecules:
                        - **Green**: Value is within ideal range for drug-like molecules
                        - **Orange**: Value is acceptable but not optimal
                        - **Red**: Value is outside of typical drug-like range
                        
                        **Note on scaling:**
                        - MW is shown divided by 100 for better visualization
                        - TPSA is shown divided by 10 for better visualization
                        
                        Annotations show the typical cutoff values for drug-like molecules.
                        """)
            else:
                st.warning("Property calculation not available for invalid molecules")

# Add a download section for the results
if st.session_state.generated_molecules:
    st.markdown('<h2 class="section-header">Export Results</h2>', unsafe_allow_html=True)
    
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
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Show preview of what will be downloaded
        with st.expander("Preview export data"):
            st.dataframe(export_df, use_container_width=True)
    
    with col2:
        # Download button with improved styling
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Results as CSV",
            data=csv,
            file_name="generated_molecules.csv",
            mime="text/csv",
            help="Download all molecules and their properties as a CSV file for further analysis",
            use_container_width=True
        )

# Add enhanced footer with usage instructions
st.markdown("---")
st.markdown('<h3 class="section-header">About This Tool</h3>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **AI-Powered Molecular Generation Pipeline** combines:
    1. Natural language understanding for target specification
    2. VAE-based molecule generation and refinement
    3. Hypergrammar validation to ensure chemical feasibility
    4. Property calculation and visualization
    """)

with col2:
    st.markdown("""
    **How to use this tool:**
    1. Enter your Gemini API key in the sidebar
    2. Describe your target molecule and set property constraints
    3. Click "Generate Molecules" to start the pipeline
    4. Explore the generated molecules in the tabs above
    5. Download results for further analysis
    """)

# Add acknowledgements in an expander
with st.expander("Technical Details & Acknowledgements"):
    st.markdown("""
    This pipeline uses:
    - **Google Gemini** for natural language understanding and initial molecule generation
    - **Variational Autoencoder (VAE)** with Symbolic Regression guidance for molecule refinement
    - **RDKit** for cheminformatics operations and property calculations
    - **Plotly** and **py3Dmol** for interactive visualizations
    
    The molecular optimization algorithm uses a latent space exploration approach with property-guided sampling to find molecules that satisfy the specified constraints while maintaining chemical feasibility.
    
    Created by Siemens RPD DAI.
    """)