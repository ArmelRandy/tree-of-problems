from typing import List
import warnings


def breakdown(terms: List[str], n_splits: int) -> List[List[str]]:
    """
    Divide a list of L element into n_splits with the following:
    - b = L // n_splits
    - r = L % n_splits, by definition  0 <= r < n_splits
    The breakdown function creates r splits with (b + 1) elements and (n_splits - r) with b elements.
    Arguments
    ---------
        terms: List[str],
            List we would like to decompose.
        n_splits: int,
            Number of components in which to decompose.
    """
    b = len(terms) // n_splits
    r = len(terms) % n_splits
    if b == 0:
        warnings.warn(
            f"The number of splits ({n_splits}) is to big. We set it to {len(terms)}. If you're okay with this, feel free to ignore the warning. Otherwise, set n_splits to a smaller value."
        )
        return [[term] for term in terms]

    left = [terms[i : i + b + 1] for i in range(0, r * (b + 1), b + 1)]
    right = [terms[i : i + b] for i in range(r * (b + 1), len(terms), b)]
    return left + right


def z(exp):
    """
    Boolean expressions
    """
    stack = []
    intervals = {}
    G = {}
    roots = []
    for i in range(len(exp)):
        if exp[i] == "(":
            G[i] = []
            if len(stack) == 0:
                roots.append(i)
                pass
            else:
                for j in stack:
                    G[j].append(i)
            stack.append(i)
        elif exp[i] == ")":
            j = stack.pop()
            intervals[j] = i
        else:
            pass
    return G, intervals, roots


PROBLEM_TO_FILENAME = {
    "boolean_expressions": "boolean_expressions",
    "dyck_languages": "dyck_languages",
    "hyperbaton": "hyperbaton",
    "logical_deduction_three_objects": "logical_deduction",
    "logical_deduction_five_objects": "logical_deduction",
    "logical_deduction_seven_objects": "logical_deduction",
    "multistep_arithmetic_two": "multistep_arithmetic_two",
    "navigate": "navigate",
    "object_counting": "object_counting",
    "sports_understanding": "sports_understanding",
    "temporal_sequences": "temporal_sequences",
    "tracking_shuffled_objects_three_objects": "tracking_shuffled_objects",
    "tracking_shuffled_objects_five_objects": "tracking_shuffled_objects",
    "tracking_shuffled_objects_seven_objects": "tracking_shuffled_objects",
    "web_of_lies": "web_of_lies",
    "word_sorting": "word_sorting",
}

BBH_TASKS = list(PROBLEM_TO_FILENAME.keys())
