# AI-Powered Molecular Generation Pipeline

## Overview

This repository, **MoleculeForge**, is an AI-powered prototype for molecular generation, optimization, and validation. It integrates cutting-edge AI techniques such as Variational Autoencoders (VAE), Symbolic Regression (SR), and Large Language Models (LLMs) to generate novel molecular structures that meet specific property constraints.

## Key Features

- **Molecular Design from Text**: Generate molecules from natural language descriptions.
- **Optimization with Constraints**: Optimize molecules for specific properties like LogP, QED, molecular weight, etc.
- **Chemical Validation**: Validate generated structures using chemical and pharmaceutical rules.
- **Visualization Tools**: Visualize molecules in 2D and 3D representations.
- **Property Calculation**: Compute relevant molecular properties for drug discovery.

## Pipeline Stages

### 1. Molecular Design Input
- Users set property constraints such as LogP range, molecular weight, and QED thresholds.
- Advanced options include the number of molecules to generate and VAE refinement iterations.
- Molecule descriptions are provided in natural language.

### 2. AI-Driven Molecular Generation
- Utilizes Google Gemini for natural language understanding to generate initial molecular structures.
- Real-time evaluation ensures chemical feasibility and filters unsuitable candidates.

### 3. VAE Refinement
- Variational Autoencoder explores chemical spaces to optimize molecules.
- Symbolic Regression guides optimization towards desired molecular property constraints.
- Iterative improvements ensure realistic and chemically valid structures.

### 4. Chemical Validation
- Validates refined molecules against chemical rules:
  - Proper valence states
  - Realistic bond configurations
  - Stable ring structures
- Marks molecules as "Valid" or "Invalid" with detailed explanations of any issues.

### 5. Property Analysis and Visualization
- Calculates and visualizes molecular properties:
  - Lipinski's Rule of Five compliance indicators
  - Interactive radar charts and property profiles
- Provides export options for datasets, structures, and visualization charts.

## Components

- `vae_model.py`: Implements the VAE model for molecular generation.
- `hypergrammar_validator.py`: Defines chemical validation rules.
- `property_calculator.py`: Computes molecular properties.
- `molecular_generator.py`: Interfaces with the Gemini API for text-to-molecule generation.
- `visualization.py`: Provides 2D and 3D visualization functions.
- `main.py`: The Streamlit application for the user interface.

## Requirements

- Python 3.8+
- RDKit
- PyTorch
- Streamlit
- py3Dmol
- Google Generative AI (Gemini API)
- gplearn
- scikit-learn
- matplotlib

## Setup and Installation

1. Clone this repository.
2. Install the required packages using `pip install -r requirements.txt`.
3. Obtain a Gemini API key from Google.
4. Run the Streamlit app using `streamlit run main.py`.

## Usage

1. Enter a natural language description of the desired molecule.
2. Set property constraints (e.g., LogP, molecular weight, QED).
3. Click "Generate Molecules" to initiate the pipeline.
4. View and export the generated molecules.

## Technical Details

- **AI Models**: Integrates Google's Gemini for text interpretation and a VAE for molecular optimization.
- **Validation**: Ensures chemical feasibility through hypergrammar validation.
- **Visualization**: Uses RDKit, py3Dmol, and Plotly for interactive molecular representations.

## Acknowledgements

This pipeline leverages:
- **Google Gemini** for natural language understanding and molecule generation.
- **Variational Autoencoder (VAE)** with Symbolic Regression for optimization.
- **RDKit** for cheminformatics operations and property calculations.

## License

This project is proprietary and confidential. All rights reserved.