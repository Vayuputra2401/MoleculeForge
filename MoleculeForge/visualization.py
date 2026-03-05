import io
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from PIL import Image

def visualize_molecule_2d(smiles, size=(400, 300), highlight_atoms=None):
    """
    Create a 2D visualization of a molecule
    
    Args:
        smiles (str): SMILES string of the molecule
        size (tuple): Size of the image (width, height)
        highlight_atoms (list): List of atom indices to highlight
        
    Returns:
        PIL.Image: 2D visualization of the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Return a blank image with text if the molecule is invalid
        img = Image.new('RGB', size, color='white')
        plt.figure(figsize=(size[0]/100, size[1]/100))
        plt.text(0.5, 0.5, "Invalid Molecule", horizontalalignment='center', 
                 verticalalignment='center', fontsize=18)
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        return img
    
    # Generate 2D coordinates for the molecule
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    mol = Chem.RemoveHs(mol)
    
    # Set drawing options
    drawer = Draw.MolDraw2DCairo(size[0], size[1])
    drawer.SetFontSize(0.8)
    
    # Draw molecule with or without highlighting
    if highlight_atoms:
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms)
    else:
        drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    
    # Convert the drawer's image to a PIL Image
    png_data = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(png_data))
    
    return img

def visualize_molecule_3d(smiles, width=400, height=400):
    """
    Create a 3D visualization of a molecule using py3Dmol
    
    Args:
        smiles (str): SMILES string of the molecule
        width (int): Width of the 3D view
        height (int): Height of the 3D view
        
    Returns:
        py3Dmol.view: 3D visualization view of the molecule
    """
    import py3Dmol
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Generate 3D coordinates for the molecule
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
    except:
        return None
    
    # Convert to mol block for py3Dmol
    mol_block = Chem.MolToMolBlock(mol)
    
    # Create py3Dmol view
    view = py3Dmol.view(width=width, height=height)
    view.addModel(mol_block, "mol")
    
    # Style the molecule
    view.setStyle({'stick': {}})
    view.setStyle({'atom': 'C'}, {'color': 'gray'})
    view.setStyle({'atom': 'O'}, {'color': 'red'})
    view.setStyle({'atom': 'N'}, {'color': 'blue'})
    view.setStyle({'atom': 'S'}, {'color': 'yellow'})
    view.setStyle({'atom': 'Cl'}, {'color': 'green'})
    view.setStyle({'atom': 'F'}, {'color': 'green'})
    view.setStyle({'atom': 'Br'}, {'color': 'brown'})
    view.setStyle({'atom': 'I'}, {'color': 'purple'})
    
    # Set up the view
    view.zoomTo()
    
    return view

def create_molecule_grid(smiles_list, per_row=3, molsPerRow=3, subImgSize=(200, 200)):
    """
    Create a grid of 2D molecule visualizations
    
    Args:
        smiles_list (list): List of SMILES strings
        per_row (int): Number of molecules per row
        molsPerRow (int): Alternative name for per_row (for compatibility)
        subImgSize (tuple): Size of each molecule image
        
    Returns:
        PIL.Image: Grid of molecule visualizations
    """
    # Handle empty lists
    if not smiles_list:
        return None
    
    # Parse molecules
    mols = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mols.append(mol)
    
    # Handle case where no valid molecules
    if not mols:
        return None
    
    # Use the Draw.MolsToGridImage function
    try:
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=per_row,
            subImgSize=subImgSize,
            legends=[f"Mol {i+1}" for i in range(len(mols))]
        )
        return img
    except:
        # Fallback: create individual images and stitch them together
        images = []
        for smiles in smiles_list:
            img = visualize_molecule_2d(smiles, size=subImgSize)
            images.append(img)
        
        # Calculate grid dimensions
        n_cols = per_row
        n_rows = (len(images) + n_cols - 1) // n_cols  # Ceiling division
        
        # Create a blank canvas
        grid_width = n_cols * subImgSize[0]
        grid_height = n_rows * subImgSize[1]
        grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
        
        # Place images in grid
        for i, img in enumerate(images):
            row = i // n_cols
            col = i % n_cols
            x = col * subImgSize[0]
            y = row * subImgSize[1]
            grid_img.paste(img, (x, y))
        
        return grid_img

def visualize_property_radar(properties, figsize=(8, 8)):
    """
    Create a radar chart of molecular properties
    
    Args:
        properties (dict): Dictionary of molecular properties
        figsize (tuple): Size of the figure
        
    Returns:
        matplotlib.figure.Figure: Radar chart of properties
    """
    # Select properties for radar chart and normalize them
    radar_props = {}
    
    # Add properties if they exist in the input dictionary
    if 'logP' in properties:
        # Normalize logP to be in [0,1] range (typical drug logP is -2 to 6)
        logp_norm = (max(min(properties['logP'], 6), -2) + 2) / 8
        radar_props['LogP'] = logp_norm
    
    if 'QED' in properties:
        # QED is already in [0,1] range
        radar_props['QED'] = properties['QED']
    
    if 'MolecularWeight' in properties:
        # Normalize MW (typical drug MW is 100-700)
        mw_norm = min(properties['MolecularWeight'], 700) / 700
        radar_props['MW'] = mw_norm
    
    if 'TPSA' in properties:
        # Normalize TPSA (typical range 0-200)
        tpsa_norm = min(properties['TPSA'], 200) / 200
        radar_props['TPSA'] = tpsa_norm
    
    if 'NumHBondDonors' in properties:
        # Normalize HBD (typically 0-5 for drugs)
        hbd_norm = min(properties['NumHBondDonors'], 5) / 5
        radar_props['HBD'] = hbd_norm
    
    if 'NumHBondAcceptors' in properties:
        # Normalize HBA (typically 0-10 for drugs)
        hba_norm = min(properties['NumHBondAcceptors'], 10) / 10
        radar_props['HBA'] = hba_norm
    
    if 'FractionCSP3' in properties:
        # FractionCSP3 is already in [0,1] range
        radar_props['Fsp3'] = properties['FractionCSP3']
    
    # If we have no properties, return None
    if not radar_props:
        return None
    
    # Create radar chart
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)
    
    # Set the angles of the plot
    angles = np.linspace(0, 2*np.pi, len(radar_props), endpoint=False).tolist()
    values = list(radar_props.values())
    
    # Complete the loop
    values += values[:1]
    angles += angles[:1]
    
    # Plot data
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    
    # Set labels
    ax.set_thetagrids(np.degrees(angles[:-1]), list(radar_props.keys()))
    
    # Set y-axis limits
    ax.set_ylim(0, 1.0)
    
    return fig
