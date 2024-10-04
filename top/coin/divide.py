from typing import List
from sentence_splitter import SentenceSplitter


def divide_fn(prompts: List[str], n_splits: int) -> List[List[str]]:
    splitter = SentenceSplitter(language="en")
    list_of_subproblems = []
    for prompt in prompts:
        # coin
        sentences = splitter.split(text=prompt)
        begin, end = sentences[0], sentences[-1]
        if n_splits == -1:
            problems = sentences[1:-1]
        else:
            sentences = sentences[1:-1]
            b = len(sentences) // n_splits
            r = len(sentences) % n_splits
            left = [
                " ".join(sentences[i : i + b + 1]) for i in range(0, r * (b + 1), b + 1)
            ]
            right = [
                " ".join(sentences[i : i + b])
                for i in range(r * (b + 1), len(sentences), b)
            ]
            problems = left + right
        subproblems = []
        for i, problem in enumerate(problems):
            subproblem = begin + " " + problem + " " + end
            if i > 0:
                subproblem = subproblem.replace("heads", "<status>").replace(
                    "tails", "<status>"
                )
            subproblems.append(subproblem)
        list_of_subproblems.append(subproblems)
    return list_of_subproblems


def divide_l2m(prompts: List[str]) -> List[List[str]]:
    splitter = SentenceSplitter(language="en")
    list_of_subproblems = []
    for prompt in prompts:
        # coin
        sentences = splitter.split(text=prompt)
        begin, end = sentences[0], sentences[-1]
        subproblems = [f"{sentences[1]} {sentences[2]}"]
        for sentence in sentences[3:-1]:
            subproblem = subproblems[-1]
            subproblem += f" {sentence}"
            subproblems.append(subproblem)
        subproblems = [
            begin + " " + subproblem + " " + end for subproblem in subproblems
        ]
        list_of_subproblems.append(subproblems)
    return list_of_subproblems
