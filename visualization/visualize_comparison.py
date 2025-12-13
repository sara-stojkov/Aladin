import plotly.express as px
import plotly.io as pio
import pandas as pd
import matplotlib.pyplot as plt

# use a browser renderer to avoid nbformat dependency in script runs
pio.renderers.default = "browser"

def visualize_comparison(df: pd.DataFrame, x_col: str, y_col: str, category_col: str = None, title: str = "Comparison Visualization", renderer: str = "browser", save_path: str = None):
    """
    Visualizes the comparison between two columns in a DataFrame using scatter plot or line plot.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to visualize.
    x_col (str): The column name for the x-axis.
    y_col (str): The column name for the y-axis.
    category_col (str, optional): The column name for categorizing data points. Defaults to None.
    title (str): The title of the plot.
    renderer (str): Plotly renderer; default "browser" avoids nbformat requirement.

    Returns:
    None: Displays the plot.
    """
    if category_col:
        fig = px.scatter(df, x=x_col, y=y_col, color=category_col, title=title)
    else:
        fig = px.scatter(df, x=x_col, y=y_col, title=title)

    fig.show(renderer=renderer)
    if save_path:
        fig.write_html(save_path)

def plot_differences_interactive(df: pd.DataFrame, x_col: str, y_col: str, hue_col: str, hover_cols: list = None, title: str = "Differences Visualization", renderer: str = "browser", save_path: str = None):
    """
    Plots the differences between two columns in a DataFrame interactively.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to visualize.
    x_col (str): The column name for the x-axis.
    y_col (str): The column name for the y-axis.
    hue_col (str): The column name for categorizing data points.
    title (str): The title of the plot.
    renderer (str): Plotly renderer; default "browser" avoids nbformat requirement.

    Returns:
    None: Displays the interactive plot.
    """

    fig = px.scatter(df, x=x_col, y=y_col, color=hue_col, title=title, hover_data=hover_cols if hover_cols is not None else df.columns)
    fig.show(renderer=renderer)

    if save_path:
        fig.write_html(save_path)


if __name__ == "__main__":
    from features.build_features import transform_data
    from datasets import load_dataset
    from features.pca import apply_pca_multiple_components
    from features.word_embeddings import glove

    # Load sample dataset
    df = load_dataset("yelp_review_full")['train'].to_pandas()
    df = df.sample(1000, random_state=42).reset_index(drop=True)
    df_transformed = transform_data(df, text_columns=['text'], lowercase=True, remove_stopwords=True, remove_special_chars=True, use_nltk_stemming=True, use_spacy_lemmatization=False)
    # Apply PCA for demonstration
    df_transformed = glove(df_transformed, text_columns=['text', 'text'])
    df_pca = apply_pca_multiple_components(df_transformed, embedded_text_col=['embedded_text','embedded_text'], n_components=2)
    # Visualize comparison
    # visualize_comparison(df_pca, x_col='pca_0', y_col='pca_1', title="PCA Component Comparison", save_path="visualization\\plots\\pca_comparison.html")
    # Plot differences interactively
    plot_differences_interactive(df_pca, x_col='pca_0', y_col='pca_1', hue_col='label', hover_cols=['text'], title="PCA Differences Interactive Visualization", save_path="visualization\\plots\\pca_differences.html")