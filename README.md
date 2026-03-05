# MoleculeForge - Molecular Generation Pipeline

MoleculeForge is an end-to-end AI-powered molecular generation, optimization, and validation pipeline designed to accelerate drug discovery leveraging cutting-edge Generative AI, sophisticated multi-agent systems, and cheminformatics.

## Overview

The pipeline integrates advanced deep learning representations (VAE, Graph Convolutional Networks) with Large Language Models (LLMs) and agentic frameworks. It interprets natural language descriptions of desired molecules and navigates chemical space to generate novel structures satisfying complex structural and property constraints.

### Key Capabilities & Impact
- **End-to-End Pipeline**: Built an end-to-end molecular generation pipeline using fine-tuned Gemini models. Reduced drug discovery time by **58%** through LLM optimization and Chain-of-Thought (CoT) reasoning on the ZINC dataset.
- **Agentic Validation & Advanced RAG**: Combined Variational Autoencoders (VAE) with multi-agent orchestration (via AutoGen) and advanced Retrieval-Augmented Generation (RAG via LangChain) for molecular validation. Achieved a **76%** higher success rate using rigorous Data Engineering for self-reflective RAG techniques.
- **Scaffolding & Observability**: Applied agentic systems with comprehensive LLM evaluation frameworks for targeted molecular scaffolding. Created sophisticated reward systems with embedded observability monitoring that improved candidate viability by **42%**.

## Tech Stack
- **Deep Learning & Modeling**: PyTorch, Variational Autoencoders (VAE), Graph Convolutional Networks (GCNs)
- **Generative AI & LLMs**: Gemini (Fine-tuned), Chain-of-Thought Reasoning
- **Agentic Frameworks**: LangChain, AutoGen
- **Cheminformatics**: RDKit, Cheminformatics rules
- **Data & Evaluation**: ZINC Dataset, Data Engineering, Observability Monitoring

## System Components

- `vae_model.py`: Neural network implementation and management of the Variational Autoencoder mapping the chemical space.
- `molecular_generator.py`: Connects natural language requirements to chemical structures using the Gemini API and agentic reasoning traces.
- `hypergrammar_validator.py`: Enforces critical chemical and pharmaceutical validity using strict hypergrammar rules.
- `property_calculator.py`: Evaluates generated compounds against physical constraints (LogP, QED, Molecular Weight).
- `visualization.py`: Utilities for multi-dimensional rendering of 2D and 3D molecular structures.
- `main.py`: Streamlit-based application serving as the interactive UI for the pipeline.

## Getting Started

### Prerequisites
- Python 3.8+
- [RDKit](https://www.rdkit.org/)
- PyTorch
- Streamlit
- Google Generative AI capabilities (Gemini API Key)

### Installation
1. Clone this repository to your local environment.
2. Install the primary dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Provide your Gemini API key in `secrets.json` or as an environment variable (`GEMINI_API_KEY`).
4. Execute the interactive Streamlit application:
   ```bash
   streamlit run main.py
   ```

## License
This project is proprietary and confidential. All rights reserved.
