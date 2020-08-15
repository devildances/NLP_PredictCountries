import numpy as np

def cosine_similarity(A, B):
    '''
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        cos: numerical number representing the cosine similarity between A and B.
    '''
    dot = np.dot(A,B)
    norma = np.linalg.norm(A)
    normb = np.linalg.norm(B)
    cos = dot/(norma*normb)
    return cos

def euclidean(A, B):
    """
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        return: numerical number representing the Euclidean distance between A and B.
    """
    return np.linalg.norm(A-B)

def get_country(city, embeddings):
    """
    Input:
        city1: a string (the capital city of country1)
        country1: a string (the country of capital1)
        city2: a string (the capital city of country2)
        embeddings: a dictionary where the keys are words and values are their embeddings
    Output:
        countries: a dictionary with the most likely country and its similarity score
    """
    group = set(('Athens', 'Greece', city))
    city1_emb = embeddings['Athens']
    country1_emb = embeddings['Greece']
    try:
        city2_emb = embeddings[city]
    except:
        return city+" is not found/unknown as a country capital. Please Check again!"
    vec = country1_emb - city1_emb + city2_emb
    similarity = -1
    country = ''

    for word in embeddings.keys():
        if word not in group:
            word_emb = embeddings[word]
            cur_similarity = cosine_similarity(vec, word_emb)
            if cur_similarity > similarity:
                similarity = cur_similarity
                country = (word, similarity)

    return country

def get_accuracy(word_embeddings, data):
    '''
    Input:
        word_embeddings: a dictionary where the key is a word and the value is its embedding
        data: a pandas dataframe containing all the country and capital city pairs
    Output:
        accuracy: the accuracy of the model
    '''
    num_correct = 0

    for i, row in data.iterrows():
        # city1 = row['city1']
        # country1 = row['country1']
        city =  row['city2']
        country = row['country2']
        predicted_country, _ = get_country(city, word_embeddings)

        if predicted_country == country:
            num_correct += 1

    m = len(data)
    accuracy = num_correct/m
    return accuracy

def compute_pca(X, n_components=2):
    """
    Input:
        X: of dimension (m,n) where each row corresponds to a word vector
        n_components: Number of components want to keep.
    Output:
        X_reduced: data transformed in 2 dims/columns + regenerated original data
    """
    X_demeaned = X - np.mean(X, axis=0)
    covariance_matrix = np.cov(X_demeaned, rowvar=False)
    eigen_vals, eigen_vecs = np.linalg.eigh(covariance_matrix, UPLO='L')
    idx_sorted = np.argsort(eigen_vals)
    idx_sorted_decreasing = idx_sorted[::-1]
    eigen_vals_sorted = eigen_vals[idx_sorted_decreasing]
    eigen_vecs_sorted = eigen_vecs[:,idx_sorted_decreasing]
    eigen_vecs_subset = eigen_vecs_sorted[:,:n_components]
    X_reduced = np.dot(eigen_vecs_subset.T, X_demeaned.T).T

    return X_reduced

def get_vectors(embeddings, words):
    """
    Input:
        embeddings: a word
        words: a list of words
    Output:
        X: a matrix where the rows are the embeddings corresponding to the rows on the list
    """
    m = len(words)
    X = np.zeros((1, 300))
    for word in words:
        english = word
        eng_emb = embeddings[english]
        X = np.row_stack((X, eng_emb))
    X = X[1:,:]
    return X