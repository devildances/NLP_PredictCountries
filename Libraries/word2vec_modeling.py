import nltk
from gensim.models import KeyedVectors
import pickle

def w2vModel(words=[]):
    print('Start to build model based on the data entered...')
    # nltk.download('punkt')
    embeddings = KeyedVectors.load_word2vec_format('RAW/GoogleNews-vectors-negative300.bin.gz', binary = True)
    f = open('RAW/capitals.txt', 'r').read()
    set_words = set(nltk.word_tokenize(f))
    select_words = words
    word_embeddings = {}

    for w in select_words:
        set_words.add(w)

    for word in embeddings.vocab:
        if word in set_words:
            word_embeddings[word] = embeddings[word]

    pickle.dump( word_embeddings, open( "Word2Vec/word_embeddings_subset.p", "wb" ) )
    print('Word2Vec model successfully built!')