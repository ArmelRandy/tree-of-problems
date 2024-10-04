import os
import pandas as pd
from top.got import DATA_PATH


def get_dataset(description: str):
    """
    Arguments
    ---------
        - description: str,
    """
    if "sorting" in description:
        print("Sorting lists of 32 elements.")
        df = pd.read_csv(os.path.join(DATA_PATH, "sorting_032.csv"))
        return [(row["Unsorted"], row["Sorted"]) for _, row in df.iterrows()]
    elif "set_intersection" in description:
        print("Intersection lists of 32 elements.")
        df = pd.read_csv(os.path.join(DATA_PATH, "set_intersection_032.csv"))
        return [
            (f"{row['SET1']} + {row['SET2']}", row["INTERSECTION"])
            for _, row in df.iterrows()
        ]
    elif "keyword_counting" in description:
        df = pd.read_csv(os.path.join(DATA_PATH, "countries.csv"))
        return [(row["Text"], row["Countries"]) for _, row in df.iterrows()]
    else:
        raise ValueError(f"Description `{description}` is not supported by GoT.")
