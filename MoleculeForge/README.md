# AI-Powered Molecular Generation Pipeline

This repository contains an AI-powered prototype for molecular generation, optimization, and validation.

## Overview

The pipeline combines multiple AI techniques to generate novel molecular structures with specific property constraints:

1. **Natural Language Understanding**: Interprets textual descriptions of desired molecules
2. **Variational Autoencoder (VAE)**: Generates and refines molecular structures
3. **Symbolic Regression**: Optimizes molecules for desired properties
4. **Hypergrammar Validation**: Ensures chemical feasibility of generated structures
5. **Property Calculation**: Computes relevant molecular properties

## Key Features

- Generate molecules from natural language descriptions
- Optimize molecules for specific property constraints (LogP, QED, molecular weight, etc.)
- Validate molecules using chemical and pharmaceutical rules
- Visualize molecules in 2D and 3D
- Calculate important molecular properties for drug discovery

## Components

- `vae_model.py`: Implementation of the VAE model for molecular generation
- `hypergrammar_validator.py`: Chemical validation rules for generated molecules
- `property_calculator.py`: Functions to calculate molecular properties
- `molecular_generator.py`: Interface to Gemini API for text-to-molecule generation
- `visualization.py`: Functions for 2D and 3D visualization of molecules
- `main.py`: Streamlit application for the user interface

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

1. Clone this repository
2. Install the required packages using `pip install -r requirements.txt`
3. Obtain a Gemini API key from Google
4. Run the Streamlit app using `streamlit run main.py`

## Usage

1. Enter a description of the molecule you want to generate
2. Set property constraints (LogP, molecular weight, QED)
3. Click "Generate Molecules" to start the pipeline
4. View and export the generated molecules

## License

This project is proprietary and confidential. All rights reserved.
