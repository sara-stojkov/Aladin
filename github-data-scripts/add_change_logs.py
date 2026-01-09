import os
import requests
import time
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
GITHUB_API = "https://api.github.com"

token = os.getenv("GITHUB_TOKEN")

if not token:
    raise RuntimeError("GITHUB_TOKEN environment variable not set")

def add_pr_change_stats_with_token(
    df: pd.DataFrame,
    token: str,
    owner: str,
    repo: str,
    sleep_seconds: float = 0.1,
) -> pd.DataFrame:


    if "number" not in df.columns:
        raise ValueError("DataFrame must contain a 'number' column")

    df = df.copy()
    df["additions"] = pd.NA
    df["deletions"] = pd.NA
    df["commits"] = pd.NA

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "pr-stats-script",
    }

    for idx, pr_number in df["number"].items():
        r = requests.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}",
            headers=headers,
        )
        r.raise_for_status()
        pr = r.json()

        if pr.get("merged_at") is None:
            continue

        df.at[idx, "additions"] = pr["additions"]
        df.at[idx, "deletions"] = pr["deletions"]
        df.at[idx, "commits"] = pr["commits"]

    return df

def load_csv_to_dataframe(csv_path):
    """
    Load a CSV file into a pandas DataFrame.
    """
    df = pd.read_csv(csv_path)
    return df


def main():
    # Path to your CSV file
    csv_file = "merged_pull_requests.csv"

    df1 = load_csv_to_dataframe(csv_file)

    # Display basic info
    df = add_pr_change_stats_with_token(
    df=df1,
    token=token,
    owner="jts",
    repo="sga",
    )  

    print("DataFrame preview:")
    print(df.head())

    print("\nDataFrame info:")
    print(df.info())

    df.to_csv("pr_change.csv", index=False)
 

if __name__ == "__main__":
    main()
    
