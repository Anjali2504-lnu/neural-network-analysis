# Neural Network Analysis

This repository contains a Jupyter Notebook that implements an image captioning pipeline using a pretrained CLIP image encoder and GPT-2 language model. The project explores how visual embeddings can be injected into a language model to generate descriptive captions for images.

## Repository Content

- `neural_network_notebook.ipynb` — the main notebook containing the full workflow, including model setup, data preparation, caption generation, training, and evaluation

## Project Overview

The notebook presents an end-to-end deep learning pipeline for image caption understanding and generation. It combines computer vision and natural language processing by extracting image features with CLIP and using GPT-2 to generate text captions from those features.

## What the Notebook Includes

- installation and setup of required libraries
- loading pretrained CLIP and GPT-2 models
- downloading and preparing COCO 2017 caption data
- extracting CLIP image embeddings
- implementing prefix-based embedding injection for caption generation
- defining an alternative cross-attention based injection method for comparison
- generating captions using greedy decoding, beam search, and nucleus sampling
- training the projection layer on sample image-caption pairs
- evaluating generated captions using BLEU, METEOR, ROUGE-L, and CIDEr
- qualitative analysis of caption quality and decoding behavior
- comparison of decoding strategies, temperature sensitivity, beam width, and fine-tuning approaches

## Tools and Libraries

The notebook is implemented in Python using Jupyter Notebook and includes libraries such as PyTorch, Transformers, NumPy, Pandas, Matplotlib, NLTK, and evaluation packages for captioning metrics.

## Usage

Clone the repository, open the notebook in Jupyter, and run the cells in sequence to reproduce the full workflow and analysis.

## Author

Anjali
