# Word2Vec Similarity Demo

This project demonstrates training a Word2Vec model on a small text corpus and querying word similarities.

## Overview

A Word2Vec model is trained on a short set of example sentences related to cats and dogs.  
The goal is to illustrate how word embeddings capture contextual similarity.

## Dataset

The model is trained on the following sentences:

- "The cat sat on the mat."
- "The dog barked at the cat."
- "Cats and dogs are great pets."
- "I love my cat."
- "Dogs are loyal animals."

Plural forms are normalized during preprocessing to improve embedding quality.

## Functionality

- Train a Word2Vec model using the skip-gram algorithm
- Retrieve the vector representation of the word **cat**
- Compute the top 5 most similar words to **cat** using cosine similarity

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the training script:

```bash
python train_word2vec.py
```

## Output

The script will output:

- The dimensionality and vector values for the word cat
- The top 5 most similar words to cat with similarity scores

## Dependencies

- gensim: For Word2Vec model training
