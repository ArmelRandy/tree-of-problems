import os
import pandas as pd
from top.coin import DATA_PATH


def get_dataset(dataset_name_or_path):
    df = pd.read_csv(os.path.join(DATA_PATH, dataset_name_or_path))
    questions = [example["Question"] for _, example in df.iterrows()]
    answers = [example["Answer"] for _, example in df.iterrows()]
    return [(question, answer) for (question, answer) in zip(questions, answers)]
