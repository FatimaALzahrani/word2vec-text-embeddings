from gensim.models import Word2Vec
import numpy as np

sentences = [
    "The cat sat on the mat.",
    "The dog barked at the cat.",
    "Cats and dogs are great pets.",
    "I love my cat.",
    "Dogs are loyal animals."
]

# Preprocess
tokenized_sentences = [sentence.lower().replace(".", "").split() for sentence in sentences]

print("Training Word2Vec model...")
print("Tokenized sentences:")
for i, tokens in enumerate(tokenized_sentences, 1):
    print(f"  {i}. {tokens}")
print()

# Train Word2Vec model
model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1,  # Using skip-gram algorithm
    epochs=100
)



# retrieve and print the vector representation of "cat"
cat_vector = model.wv['cat']
print(f"Dimensionality: {len(cat_vector)}")
print(f"Vector: {cat_vector}")
print()

# find and print the top 5 most similar words to "cat"
try:
    similar_words = model.wv.most_similar('cat', topn=5)
    for rank, (word, similarity) in enumerate(similar_words, 1):
        print(f"  {rank}. {word:15s} (similarity: {similarity:.4f})")
except Exception as e:
    print(f"{e}")


print(f"Total words in vocabulary: {len(model.wv)}")
print(f"Words: {list(model.wv.index_to_key)}")
