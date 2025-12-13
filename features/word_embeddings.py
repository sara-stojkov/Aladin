import pandas as pd

def bag_of_words(df: pd.DataFrame, max_features: int = None, return_matrices: bool = False, text_columns: list = ['code_before', 'code_after']):
    """
    Bag of Words representation using `CountVectorizer`.

    :param df: DataFrame with `review_title` and `review_body` columns
    :param max_features: cap vocabulary size to avoid huge matrices
    :param return_matrices: if True, return `(df, X_title, X_body)` sparse matrices

    Behavior:
    - By default this function returns `df` unchanged but will attach nothing to it
      for large corpora. If the caller needs the document-term matrices, pass
      `return_matrices=True` to get the sparse matrices back and avoid implicit
      dense conversions.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer_rt = CountVectorizer(max_features=max_features)
    X_rt = vectorizer_rt.fit_transform(df[text_columns[0]].fillna(''))

    vectorizer_rb = CountVectorizer(max_features=max_features)
    X_rb = vectorizer_rb.fit_transform(df[text_columns[1]].fillna(''))

    # If caller asked for the raw sparse matrices, return them alongside the dataframe
    if return_matrices:
        return df, X_rt, X_rb

    # Backwards-compatible behaviour: store dense vectors in dataframe cells only
    # when safe. Guard against creating an enormous dense matrix.
    n_rows, n_features = X_rb.shape
    if n_rows * n_features > 30_000_000:
        raise MemoryError(
            f"Bag-of-words dense conversion would create {n_rows * n_features} elements; "
            "call bag_of_words(..., return_matrices=True) instead or set max_features to a smaller value"
        )

    df[f'embedded_{text_columns[0]}'] = list(X_rt.toarray())
    df[f'embedded_{text_columns[1]}'] = list(X_rb.toarray())
    return df

def tf_idf(df: pd.DataFrame, max_features: int = None, return_matrices: bool = False, text_columns: list = ['code_before', 'code_after']):
    """
    TF-IDF (Term Frequency-Inverse Document Frequency) is a text representation technique that reflects the importance of a word in a document relative to a collection of documents (corpus).
    It consists of two components: Term Frequency (TF) and Inverse Document Frequency (IDF).
    TF measures how frequently a word appears in a document, while IDF measures how important a word is by reducing the weight of common words and increasing the weight of rare words.

    :param df: DataFrame with review_title and review_body columns
    :param max_features: Maximum number of features (vocabulary size) to consider. If None, all features are used.
    :param return_matrices: If True, returns the TF-IDF matrices along with the dataframe.

    :return: returns TF-IDF feature matrices and vocabularies
    """

    from sklearn.feature_extraction.text import TfidfVectorizer

    # allow controlling vocabulary size to avoid exploding feature matrices
    vectorizer_rt = TfidfVectorizer(max_features=max_features)
    X_rt = vectorizer_rt.fit_transform(df[text_columns[0]].fillna(''))

    vectorizer_rb = TfidfVectorizer(max_features=max_features)
    X_rb = vectorizer_rb.fit_transform(df[text_columns[1]].fillna(''))

    # If caller asked for the raw matrices, return them alongside the dataframe
    if return_matrices:
        return df, X_rt, X_rb

    # Backwards-compatible behaviour: store dense vectors in dataframe cells.
    # Guard against attempting to create a huge dense array.
    n_rows, n_features = X_rb.shape
    # threshold = ~30 million elements (~240 MB for float64) — adjustable
    if n_rows * n_features > 30_000_000:
        raise MemoryError(
            f"TF-IDF dense conversion would create {n_rows * n_features} elements; "
            "call tf_idf(..., return_matrices=True) instead or set max_features to a smaller value"
        )

    df[f'embedded_{text_columns[0]}'] = list(X_rt.toarray())
    df[f'embedded_{text_columns[1]}'] = list(X_rb.toarray())

    return df

def word2vec(df: pd.DataFrame, cbow: bool = True, vector_size: int = 100, window: int = 5, min_count: int = 1, epochs: int = 5, text_columns: list = ['code_before', 'code_after']):
    """
    Word2Vec is a popular word embedding technique that represents words as dense vectors in a continuous vector space.
    It contains two main architectures: Continuous Bag of Words (CBOW) and Skip-Gram.
    CBOW predicts a target word based on its surrounding context words, while Skip-Gram predicts context words given a target word.

    CBOW is efficient for smaller datasets and captures semantic relationships well.
    Skip-Gram is particularly effective for capturing semantic relationships between words, especially for infrequent words. 

    :param df: Description
    :param cbow: Whether to use Continuous Bag of Words (CBOW) architecture. If False, Skip-Gram is used.
    :param vector_size: Dimensionality of the word vectors.
    :param window: Maximum distance between the current and predicted word within a sentence.
    :param min_count: Ignores all words with total frequency lower than this.
    :param epochs: Number of iterations (epochs) over the corpus.

    :return: returns trained Word2Vec model
    """
    from gensim.models import Word2Vec

    # different implementations for cbow and skip-gram
    if cbow:
        cbow_model = Word2Vec(sentences=[text.split() for text in df[text_columns[1]].fillna('')], vector_size=vector_size, window=window, min_count=min_count, sg=0, epochs=epochs)
        cbow_model.train([text.split() for text in df[text_columns[1]].fillna('')], total_examples=len(df[text_columns[1]].fillna('')), epochs=epochs)
        
        df[f'embedded_{text_columns[1]}'] = df[text_columns[1]].apply(lambda x: [cbow_model.wv[word] for word in x.split() if word in cbow_model.wv] if isinstance(x, str) else [])
        df[f'embedded_{text_columns[0]}'] = df[text_columns[0]].apply(lambda x: [cbow_model.wv[word] for word in x.split() if word in cbow_model.wv] if isinstance(x, str) else [])
        
        return df
    
    else:
    # if not cbow, use skip-gram
        from nltk.tokenize import word_tokenize
        import nltk
        nltk.download('punkt')

        tokens_rb = [word_tokenize(text) for text in df[text_columns[1]].fillna('')]
        skip_gram_model = Word2Vec(sentences=tokens_rb, vector_size=vector_size, window=window, min_count=min_count, sg=1, epochs=epochs)
        skip_gram_model.train(tokens_rb, total_examples=len(tokens_rb), epochs=epochs)

        df[f'embedded_{text_columns[1]}'] = df[text_columns[1]].apply(lambda x: [skip_gram_model.wv[word] for word in x.split() if word in skip_gram_model.wv] if isinstance(x, str) else [])
        df[f'embedded_{text_columns[0]}'] = df[text_columns[0]].apply(lambda x: [skip_gram_model.wv[word] for word in x.split() if word in skip_gram_model.wv] if isinstance(x, str) else [])

        return df


def glove(df: pd.DataFrame = None, text_columns: list = ['code_before', 'code_after']):
    """
    GloVe (Global Vectors for Word Representation) is trained on global word co-occurrence statistics. It leverages the global context to create word embeddings that reflect the overall meaning of words based on their co-occurrence probabilities.
    It is used to check similarity between words, perform word analogies, and as input features for various NLP tasks.
    
    :return: GloVe pre-trained model
    """
    from gensim.downloader import load

    glove_model = load("glove-wiki-gigaword-100")

    df[f'embedded_{text_columns[1]}'] = df[text_columns[1]].apply(lambda x: [glove_model[word] for word in x.split() if word in glove_model] if isinstance(x, str) else [])
    df[f'embedded_{text_columns[0]}'] = df[text_columns[0]].apply(lambda x: [glove_model[word] for word in x.split() if word in glove_model] if isinstance(x, str) else [])
    return df

def fasttext(df: pd.DataFrame = None, text_columns: list = ['code_before', 'code_after']):
    """
    FastText is an extension of Word2Vec that represents words as bags of character n-grams. It is a pre-trained model.
    This allows FastText to generate embeddings for out-of-vocabulary words by composing them from their n-grams.
    It is particularly useful for morphologically rich languages and handling rare or misspelled words.

    :return: FastText pre-trained model
    """
    import gensim.downloader as api
    fasttext_model = api.load("fasttext-wiki-news-subwords-300")
    df[f'embedded_{text_columns[1]}'] = df[text_columns[1]].apply(lambda x: [fasttext_model[word] for word in x.split() if word in fasttext_model] if isinstance(x, str) else [])
    df[f'embedded_{text_columns[0]}'] = df[text_columns[0]].apply(lambda x: [fasttext_model[word] for word in x.split() if word in fasttext_model] if isinstance(x, str) else [])

    return df

def bert(df: pd.DataFrame = None, text_columns: list = ['code_before', 'code_after']):
    """
    BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model that captures context from both directions (left and right) in a sentence. 
    It is pre-trained on large corpora using masked language modeling and next sentence prediction tasks.
    BERT embeddings are context-aware, meaning the same word can have different embeddings based on its surrounding words.

    :return: BERT pre-trained tokenizer and model
    """

    from transformers import BertModel, BertTokenizer
    import torch
    import numpy as np

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    # If no dataframe passed, return tokenizer and model so callers can use them directly
    if df is None:
        return tokenizer, bert_model

    # compute pooled embeddings in batches and attach to dataframe
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bert_model.to(device)

    texts_body = df[text_columns[1]].fillna('').astype(str).tolist()
    texts_title = df[text_columns[0]].fillna('').astype(str).tolist()

    def embed_texts(texts, batch_size: int = 32, max_length: int = 128):
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
                enc = {k: v.to(device) for k, v in enc.items()}
                out = bert_model(**enc)

                # Prefer pooler_output when available
                if hasattr(out, 'pooler_output') and out.pooler_output is not None:
                    batch_emb = out.pooler_output.cpu().numpy()
                else:
                    # fallback to mean pooling over the last hidden state using attention mask
                    last = out.last_hidden_state.cpu()
                    attn = enc['attention_mask'].cpu().unsqueeze(-1).float()
                    summed = (last * attn).sum(dim=1)
                    lens = attn.sum(dim=1).clamp(min=1)
                    batch_emb = (summed / lens).numpy()

                embeddings.append(batch_emb)

        if embeddings:
            return np.vstack(embeddings).tolist()
        return [[] for _ in texts]

    df[f'embedded_{text_columns[1]}'] = embed_texts(texts_body)
    df[f'embedded_{text_columns[0]}'] = embed_texts(texts_title)
    return df

if __name__ == "__main__":
    try:
        from datasets import load_dataset
        df = load_dataset("yelp_review_full")['train'].to_pandas()
        if df.empty:
            print('word_embedding: dataset empty — check path')
        else:
            sample = df.head(30)
            print('Running bag_of_words on a small sample')
            print(bag_of_words(sample).head())
            print('Running tf_idf on a small sample')
            print(tf_idf(sample).head())
            print('Running word2vec on a small sample')
            print(word2vec(sample).head())
            print(word2vec(sample, cbow=False).head())
            print('Running glove on a small sample')
            print(glove(sample).head())
            print('Running fasttext on a small sample')
            print(fasttext(sample).head())
            print('Running bert on a small sample')
            print(bert(sample).head())


            print('word embedding models instantiated successfully')

    except Exception as e:
        print('word_embedding module test: FAIL', e)