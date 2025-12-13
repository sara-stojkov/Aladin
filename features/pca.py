import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def _extract_vector(cell):
    """Return a 1D numpy vector for a cell that may contain:
    - a 1D numpy array (pooled embedding)
    - a 2D numpy array or list-of-token-vectors -> mean-pooled to 1D
    - a flat numeric list/tuple
    - None or empty -> returns None
    """
    if cell is None:
        return None
    if isinstance(cell, np.ndarray):
        if cell.ndim == 1:
            return cell.astype(float)
        # tokens x dim -> mean pool
        return cell.mean(axis=0).astype(float)

    if isinstance(cell, (list, tuple)):
        if len(cell) == 0:
            return None
        first = cell[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            try:
                arr = np.array(cell, dtype=float)
            except Exception:
                return None
            if arr.ndim == 1:
                return arr.astype(float)
            return arr.mean(axis=0).astype(float)
        try:
            return np.array(cell, dtype=float).astype(float)
        except Exception:
            return None

    try:
        return np.array(cell, dtype=float).ravel().astype(float)
    except Exception:
        return None


def _build_combined_embeddings(df, first_col='embedded_review_body', title_col='embedded_review_title'):
    """Return concatenated per-row vectors (N x D) built from available embedding columns.

    If both body and title exist, result is [body | title]. If only one exists, result is that column.
    """
    cols = []
    vecs = []

    if first_col in df.columns:
        first_vecs = [_extract_vector(x) for x in df[first_col]]
        cols.append(('first', first_vecs))
    else:
        first_vecs = None

    if title_col in df.columns:
        title_vecs = [_extract_vector(x) for x in df[title_col]]
        cols.append(('title', title_vecs))
    else:
        title_vecs = None

    if not cols:
        raise ValueError(f"No embedding columns found. Expected one of: {first_col}, {title_col}")

    # determine vector dim from first non-empty vector across available cols
    first_non_empty = None
    for name, vs in cols:
        for v in vs:
            if v is not None:
                first_non_empty = v
                break
        if first_non_empty is not None:
            break

    if first_non_empty is None:
        raise ValueError(f"No embeddings found in columns: {', '.join([c for c,_ in cols])}")

    base_dim = first_non_empty.shape[0]
    N = len(df)

    arrays = []
    for name, vs in cols:
        arr = np.zeros((N, base_dim), dtype=float)
        for i, v in enumerate(vs):
            if v is None:
                continue
            L = min(v.shape[0], base_dim)
            arr[i, :L] = v[:L]
        arrays.append(arr)

    if len(arrays) == 1:
        X = arrays[0]
    else:
        X = np.hstack(arrays)

    return X


def apply_pca_single_component(df: pd.DataFrame) -> pd.DataFrame:
    """Compute PCA(1) over combined review embeddings and attach `pca` column to `df`.

    Handles missing embeddings robustly and pads/truncates per-token vectors as needed.
    """
    X = _build_combined_embeddings(df)
    pca = PCA(n_components=1, random_state=42)
    values = pca.fit_transform(X).squeeze()
    out = df.copy()
    out['pca'] = values
    return out


def apply_pca_multiple_components(df: pd.DataFrame, n_components: int = 2, embedded_text_col: list = ['embedded_text', 'embedded_text']) -> pd.DataFrame:
    """Compute PCA(n_components) and attach `pca_0`, `pca_1`, ... columns to `df`.

    Returns the dataframe with added PCA component columns.
    """
    if n_components < 1:
        raise ValueError('n_components must be >= 1')
    
    X = _build_combined_embeddings(df, embedded_text_col[0], embedded_text_col[1])
    pca = PCA(n_components=n_components, random_state=42)
    comps = pca.fit_transform(X)
    out = df.copy()
    for i in range(comps.shape[1]):
        out[f'pca_{i}'] = comps[:, i]
    return out

if __name__ == "__main__":
    from datasets import load_dataset
    df = load_dataset("yelp_review_full")['train'].to_pandas()
    print(df.shape)
    print(df.head())
    print(df.columns)
    df = df.sample(1000, random_state=42).reset_index(drop=True)

    from features.word_embeddings import word2vec

    df_embedded = word2vec(df, text_columns=['text', 'text'])
    print("Embeddings added.")

    df_pca = apply_pca_multiple_components(df_embedded, n_components=2)
    print("PCA applied.")
    print(df_pca[['pca_0', 'pca_1']].head())