from typing import List


def divide_fn(prompts: List[str], n_splits: int) -> List[List[str]]:
    list_of_subproblems = []
    for prompt in prompts:
        sentences = prompt.split(", ")
        if n_splits == -1:
            list_of_subproblems.append(sentences)
        else:
            b = len(sentences) // n_splits
            r = len(sentences) % n_splits
            left = [
                ", ".join(sentences[i : i + b + 1])
                for i in range(0, r * (b + 1), b + 1)
            ]
            right = [
                ", ".join(sentences[i : i + b])
                for i in range(r * (b + 1), len(sentences), b)
            ]
            subproblems = left + right
            list_of_subproblems.append(subproblems)
    return list_of_subproblems


def divide_l2m(prompts: List[str]) -> List[List[str]]:
    list_of_subproblems = []
    for prompt in prompts:
        sentences = prompt.split(", ")
        subproblems = [f"{sentences[0]}, {sentences[1]}"]
        for sentence in sentences[2:]:
            subproblem = subproblems[-1]
            subproblem += f", {sentence}"
            subproblems.append(subproblem)
        list_of_subproblems.append(subproblems)
    return list_of_subproblems
