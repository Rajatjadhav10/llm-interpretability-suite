This repository contains two main components:

Wikipedia Dataset Builder
A command-line tool to extract, clean, and format Wikipedia articles into a ready-to-use dataset.

Attention Visualizer
A simple web application that lets you input text and see how a Transformer-based model (e.g., BERT) distributes attention across tokens.

Table of Contents
Overview

Prerequisites

Installation

Project Structure

Usage

1. Wikipedia Dataset Builder

2. Attention Visualizer

Configuration

Dataset Builder Configurations

Attention Visualizer Requirements

Adding or Updating Code

License

Overview
This project is divided into two separate tools:

Wikipedia Dataset Builder

Streams Wikipedia articles.

Cleans out markup, citations, and other unwanted text.

Optionally splits long articles into fixed-size chunks.

Outputs the results in JSONL or Parquet format.

Attention Visualizer

Runs a pre-trained Transformer model (such as BERT) on any user-provided text.

Displays a heatmap of attention weights for a chosen layer and head.

Built as a simple web app using Streamlit.

Each tool can be used independently. The dataset builder prepares clean text data, and the visualizer helps you inspect how a language model attends to that text.

Prerequisites
Python 3.8 or higher

Git (for cloning or managing this repository)

A working internet connection (to download model weights the first time you run the visualizer)

Installation
Clone this repository (if you have not already):


git clone https://github.com/Rajatjadhav10/llm-interpretability-suite.git
cd llm-interpretability-suite
(Optional) Create and activate a virtual environment:


python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
Install the requirements for the Wikipedia Dataset Builder:



cd wiki_dataset_builder
pip install -r requirements.txt
cd ..

Install the requirements for the Attention Visualizer:


cd attention-visualizer
pip install -r requirements.txt
cd ..

At this point, both tools’ dependencies will be installed in your environment.

Usage
1. Wikipedia Dataset Builder
Go to the builder folder:


cd wiki_dataset_builder
Explore a few sample articles (prints a few records to the console):


python main_explorer.py --samples 5
Run the dataset builder with a configuration. For a quick test:


python main_builder.py --config configs/test.yaml
This will process a small number of articles (as defined in test.yaml) and write output to output/test/.

To build a larger, chunked dataset for retrieval tasks:


python main_builder.py --config configs/rag.yaml
This configuration enables chunking and writes both JSONL and Parquet files to output/rag/.

For a full production run (no chunking):


python main_builder.py --config configs/production.yaml
Adjust max_articles or other settings directly in the YAML if needed.

Output files will appear under wiki_dataset_builder/output/<profile>/, for example:

wiki_default.jsonl

wiki_default.parquet

build_statistics.json

2. Attention Visualizer
Go to the visualizer folder:


cd attention-visualizer
Run the Streamlit app:


streamlit run src/app.py
In your browser, you will see a page that allows you to:

Select a pre-trained model (e.g., bert-base-uncased).

Enter or paste any sentence or short paragraph.

Choose a layer index and a head index.

Click “Visualize Attention” to view a heatmap of attention scores.

The first time you select a model, Streamlit will download the model weights. Subsequent runs will reuse the cached files.

Configuration
Dataset Builder Configurations
Configuration files for the builder are YAML files located in wiki_dataset_builder/configs/. Common options include:

name and version of the Wikipedia snapshot

streaming (true/false)

max_articles, min_article_length, max_article_length

enable_chunking, chunk_size, chunk_overlap

Cleaning options (remove citations, infoboxes, external links, etc.)

Output format (jsonl, parquet, or both)

Output directory and filename prefix

Logging level and log file path

Edit these YAML files to customize how many articles you want to process and which cleaning or chunking options to use.

Attention Visualizer Requirements
The visualizer requires:

streamlit

transformers

torch

numpy

plotly

These are listed in attention-visualizer/requirements.txt. If you wish to add more Transformer models or plotting libraries, install them with pip and update the requirements file accordingly.

Adding or Updating Code
Branching
Create a new branch for any feature or bugfix:


git checkout -b feature/your-feature-name
Staging Changes
For example, to add or update the Attention Visualizer only:

git add attention-visualizer/
Committing
Write a clear and concise commit message:

bash

git commit -m "feat: add new attention-visualizer functionality"
Pushing
Push your branch to GitHub and open a Pull Request:


git push -u origin feature/your-feature-name
Merging
After review, merge your Pull Request into main or the target branch. Delete the feature branch if no longer needed.





Project Structure. 


llm-interpretability-suite/
├── wiki_dataset_builder/          # Part 1: Dataset Builder
│   ├── src/
│   │   ├── config.py
│   │   ├── explorer.py
│   │   ├── cleaner.py
│   │   ├── chunker.py
│   │   ├── builder.py
│   │   ├── utils.py
│   │   ├── main_builder.py
│   │   └── main_explorer.py
│   ├── configs/
│   │   ├── default.yaml
│   │   ├── test.yaml
│   │   ├── rag.yaml
│   │   └── production.yaml
│   ├── output/                    # Generated datasets and statistics
│   ├── logs/                      # Builder log files
│   ├── tests/                     # Unit tests for builder components
│   ├── requirements.txt
│   └── README.md
│
├── attention-visualizer/          # Part 2: Attention Visualizer
│   ├── src/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── model_utils.py
│   │   ├── text_processing.py
│   │   ├── visualizations.py
│   │   └── app.py
│   ├── requirements.txt
│   └── README.md
│
├── docs/                          # Documentation (guides, diagrams)
├── .gitignore
├── README.md                      # This file
└── requirements-all.txt           # Combined dependencies (optional)
