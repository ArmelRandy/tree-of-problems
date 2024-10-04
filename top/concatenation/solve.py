from typing import List
from top.concatenation import PROMPTS_PATH
import os


def get_solve_prompt(prompt: str, description: str = "concatenation", k=8):
    """
    Build the few-shot prompt to solve the problem of interest
    """
    out = "What is the concatenation of the last letters of the following words?\n\n"
    out += (
        open(os.path.join(PROMPTS_PATH, f"{description}/{description}{k}.txt"), "r")
        .read()
        .strip()
        + "\n\n"
    )
    out += f"Q: {prompt}\nA:"
    return out


def cot(samples: List[str]) -> List[str]:
    """
    Returns the chain of thought of last letter concatenation
    """
    chain_of_thoughts = []
    for sample in samples:
        prompt = ""
        names = sample.split(", ")
        for name in names:
            prompt += f'The last letter of "{name}" is "{name[-1]}". '
        solutions = ", ".join(['"' + f"{name[-1]}" + '"' for name in names])
        prompt += f"Concatenating {solutions} leads to \"{''.join([name[-1] for name in names])}\"."
        prompt += (
            f" So, \"{sample}\" outputs \"{''.join([name[-1] for name in names])}\"."
        )
        chain_of_thoughts.append(prompt)
    return chain_of_thoughts
