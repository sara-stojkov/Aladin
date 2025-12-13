import pycode_similar

def detect_code_differences(referenced_code_str, candidate_code_strs):
    return pycode_similar.detect([referenced_code_str] + candidate_code_strs, diff_method=pycode_similar.UnifiedDiff, keep_prints=False, module_level=False)


if __name__ == "__main__":
    from visualization.visualize_comparison import visualize_comparison, plot_differences_interactive
    from features.build_features import transform_data
    
    from datasets import load_dataset
    import pandas as pd
    df = load_dataset("yelp_review_full")['train'].to_pandas()
    df = df.sample(1000, random_state=42).reset_index(drop=True)
    df_transformed = transform_data(df, text_columns=['text'], lowercase=True, remove_stopwords=True, remove_special_chars=True, use_nltk_stemming=True, use_spacy_lemmatization=False)
