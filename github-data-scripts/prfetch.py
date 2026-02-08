import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

GITHUB_API = "https://api.github.com"

token = os.getenv("GITHUB_TOKEN")


def get_pull_requests(owner, repo, token=None):
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    prs = []
    page = 1

    while True:
        r = requests.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/pulls",
            headers=headers,
            params={
                "state": "closed",
                "per_page": 100,
                "page": page,
            },
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            break

        prs.extend(data)
        page += 1

    return prs


def merged_prs_to_dataframe(prs):
    rows = []

    for pr in prs:
        if pr["merged_at"] is None:
            continue

        rows.append({
            "number": pr["number"],
            "title": pr["title"],
            "author": pr["user"]["login"],
            "created_at": pd.to_datetime(pr["created_at"]),
            "merged_at": pd.to_datetime(pr["merged_at"]),
            "base_branch": pr["base"]["ref"],
            "head_branch": pr["head"]["ref"],
            "url": pr["html_url"],
        })

    return pd.DataFrame(rows)


def main():
    owner = "jts"
    repo = "sga"
    token = None

    prs = get_pull_requests(owner, repo, token)
    df = merged_prs_to_dataframe(prs)

    df = df.sort_values("merged_at", ascending=False)

    print(df)
    df.to_csv("merged_pull_requests.csv", index=False)
    

if __name__ == "__main__":
    main()

