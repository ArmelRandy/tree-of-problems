from top.algebraic import DATA_PATH
import pandas as pd
import os


def get_dataset(dataset_name_or_path):
    df = pd.read_csv(os.path.join(DATA_PATH, dataset_name_or_path))
    questions = [example["Question"] for _, example in df.iterrows()]
    answers = [eval(question) for question in questions]
    return [(question, answer) for (question, answer) in zip(questions, answers)]
