import pandas as pd

del_weight = 2
commit_weight = 20

df = pd.read_csv("pr_change.csv")

df["weight"] = (
    df["additions"]
    + del_weight * df["deletions"]
    + commit_weight * df["commits"]
)

df = df.sort_values(by="weight", ascending=True)

df = df.iloc[: len(df) // 2]

df.to_csv("lEissues.csv", index=False)

#send pre-merge and post-merge code to pca

#send issue data to api
