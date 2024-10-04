import re
from typing import List


def divide_fn(prompts: List[str], n_splits: int) -> List[List[str]]:
    list_of_subproblems = []
    for prompt in prompts:
        terms = re.split(r"[\+\-]", prompt)
        # Remove any empty strings from the resulting list
        terms = [term.strip() for term in terms if term.strip()]
        signs = re.findall(r"[\+\-]", prompt)
        if len(signs) == len(terms) - 1:
            signs = ["+"] + signs
        b = len(terms) // n_splits
        r = len(terms) % n_splits
        if b == 0:
            raise ValueError(
                f"The number of splits ({n_splits}) is to big. Set it to a smaller value."
            )

        left = [terms[i : i + b + 1] for i in range(0, r * (b + 1), b + 1)]
        left_signs = [signs[i : i + b + 1] for i in range(0, r * (b + 1), b + 1)]
        right = [terms[i : i + b] for i in range(r * (b + 1), len(terms), b)]
        right_signs = [signs[i : i + b] for i in range(r * (b + 1), len(signs), b)]
        total = left + right
        total_signs = left_signs + right_signs
        subproblems = []
        for j in range(len(total)):
            expression = " ".join(
                [op + " " + val for (op, val) in zip(total_signs[j], total[j])]
            )
            expression = expression.strip()
            if expression.startswith("+"):
                expression = expression[1:].strip()
            subproblems.append(expression)
        list_of_subproblems.append(subproblems)
    return list_of_subproblems


def divide_l2m(prompts: List[str]) -> List[List[str]]:
    list_of_subproblems = []
    for prompt in prompts:
        terms = re.split(r"[\+\-]", prompt)
        # Remove any empty strings from the resulting list
        terms = [term.strip() for term in terms if term.strip()]
        signs = re.findall(r"[\+\-]", prompt)
        if len(signs) == len(terms) - 1:
            signs = ["+"] + signs
        subproblems = (
            [f"{signs[0]} {terms[0]} {signs[1]} {terms[1]}"]
            if signs[0] == "-"
            else [f"{terms[0]} {signs[1]} {terms[1]}"]
        )
        for _, (op, val) in enumerate(zip(signs[2:], terms[2:])):
            subproblem = subproblems[-1]
            subproblem += f" {op} {val}"
            subproblems.append(subproblem)
        list_of_subproblems.append(subproblems)
    return list_of_subproblems
