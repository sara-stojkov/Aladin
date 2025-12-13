import pandas as pd
import nltk
from functools import lru_cache

def stem_word(word: str) -> str:
    """
    Stems an English word.

    Parameters:
    word (str): The word to be stemmed.

    Returns:
    str: The stemmed word.
    """
    stemmer = nltk.stem.SnowballStemmer('english')
    return stemmer.stem(word)

def lemmatize_word(word: str) -> str:
    """
    Lemmatizes an English word.

    Parameters:
    word (str): The word to be lemmatized.

    Returns:
    str: The lemmatized word.
    """
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return lemmatizer.lemmatize(word)

def transform_data(df: pd.DataFrame, text_columns: list, lowercase: bool = True, remove_stopwords: bool = True, remove_special_chars: bool = True, use_nltk_stemming: bool = False, use_spacy_lemmatization: bool = False) -> pd.DataFrame:
    """
    Function that does the data cleaning and data processing and transformation for English text.

    Parameters:
    df (pd.DataFrame): The original DataFrame.
    lowercase (bool): Whether to convert text to lowercase.
    remove_stopwords (bool): Whether to remove stopwords from text.
    remove_special_chars (bool): Whether to remove special characters from text.
    use_nltk_stemming (bool): Whether to apply NLTK word stemming.
    use_spacy_lemmatization (bool): Whether to use spaCy for lemmatization.

    Returns:
    pd.DataFrame: The transformed DataFrame.
    """

    # text columns for NLP
    text_cols = text_columns

    # ensure text columns are strings
    for col in text_cols:
        df[col] = df[col].fillna('').astype(str)

    # basic cleaning first so downstream token operations see normalized text
    if remove_special_chars:
        for col in text_cols:
            df[col] = df[col].str.replace(r"[^\w\s]", ' ', regex=True)

    if lowercase:
        for col in text_cols:
            df[col] = df[col].str.lower()

    # If requested, use spaCy for lemmatization and stopword removal
    if use_spacy_lemmatization:
        try:
            import spacy
        except Exception as e:
            raise ImportError("spaCy is not installed. Install spaCy with 'pip install spacy' and download the model with 'python -m spacy download en_core_web_sm'") from e

        try:
            nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        except OSError:
            raise OSError("English spaCy model not found. Please install it with: python -m spacy download en_core_web_sm")

        for col in text_cols:
            texts = df[col].tolist()
            docs = nlp.pipe(texts, batch_size=64)
            processed = []
            for doc in docs:
                tokens = []
                for token in doc:
                    if not token.is_alpha:
                        continue
                    if remove_stopwords and token.is_stop:
                        continue
                    # use lemma
                    tok = token.lemma_.lower()
                    # apply stemming if requested
                    if use_nltk_stemming:
                        tok = stem_word(tok)
                    if tok:
                        tokens.append(tok)
                processed.append(' '.join(tokens))
            df[col] = processed
    
    elif remove_stopwords or use_nltk_stemming:
        # Use NLTK for stopword removal and/or stemming
        if remove_stopwords:
            try:
                stopwords = set(nltk.corpus.stopwords.words('english'))
            except LookupError:
                nltk.download('stopwords')
                stopwords = set(nltk.corpus.stopwords.words('english'))
        else:
            stopwords = set()

        def _process_tokens(tokens):
            out = []
            for t in tokens:
                if not t:
                    continue
                # check stopwords
                if remove_stopwords and t in stopwords:
                    continue
                # apply stemming
                if use_nltk_stemming:
                    t = stem_word(t)
                if t:
                    out.append(t)
            return ' '.join(out)

        for col in text_cols:
            df[col] = df[col].str.split().apply(_process_tokens)
        
    return df

if __name__ == "__main__":
    from datasets import load_dataset
    df = load_dataset("yelp_review_full")['train'].to_pandas()
    print(df.shape)
    print(df.head())
    print(df.columns)
    df = df.sample(1000, random_state=42).reset_index(drop=True)
    df_transformed = transform_data(df, text_columns=['text'], lowercase=True, remove_stopwords=True, remove_special_chars=True, use_nltk_stemming=True, use_spacy_lemmatization=False)
    print(df_transformed['text'].sample(frac=1, random_state=42).head(40))
    # df_transformed.to_csv("dataset_transformed.csv", index=False)