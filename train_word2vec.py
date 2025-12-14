from gensim.models import Word2Vec

sentences = [
    "The cat sat on the mat.",
    "The dog barked at the cat.",
    "Cats and dogs are great pets.",
    "I love my cat.",
    "Dogs are loyal animals."
]

def normalize_token(token: str) -> str:
    if token == "cats":
        return "cat"
    if token == "dogs":
        return "dog"
    return token

tokenized_sentences = [
    [normalize_token(w) for w in sentence.lower().replace(".", "").split()]
    for sentence in sentences
]

model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=30,
    window=5,
    min_count=1,
    sg=1,
    epochs=200
)

cat_vector = model.wv["cat"]
print(len(cat_vector))
print(cat_vector)

similar_words = model.wv.most_similar("cat", topn=5)
for word, similarity in similar_words:
    print(word, similarity)
