import re
import os
from typing import List
from top.algebraic import PROMPTS_PATH


def get_solve_prompt(prompt: str, description: str, k=8):
    """
    Build the few-shot prompt to solve the problem of interest
    """
    out = "What are the results of the following algebraic operations.\n\n"
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
    Returns the chain of thought of algebraïc sum
    """
    chain_of_thoughts = []
    for sample in samples:
        is_negative = False
        if sample.startswith("-"):
            sample = sample[1:]
            is_negative = True
        terms = re.split(r"[\+\-]", sample)
        # Remove any empty strings from the resulting list
        terms = [term.strip() for term in terms if term.strip()]
        signs = re.findall(r"[\+\-]", sample)
        prompt = "Let's think step by step. "
        starting_value = f"{'-'*is_negative}{terms[0]} {signs[0]} {terms[1]}"
        if signs[0] == "+":
            prompt += f"Add, {starting_value} = {eval(starting_value)}. "
        else:
            prompt += f"Subtract, {starting_value} = {eval(starting_value)}. "
        last_value = eval(starting_value)
        for j in range(2, len(terms)):
            value = eval(f"{last_value} {signs[j-1]} {terms[j]}")
            if signs[j - 1] == "+":
                prompt += f"Add, {last_value} {signs[j-1]} {terms[j]} = {value}. "
            else:
                prompt += f"Subtract, {last_value} {signs[j-1]} {terms[j]} = {value}. "
            last_value = value
        prompt += f"So the answer is {value}."
        chain_of_thoughts.append(prompt)
    return chain_of_thoughts
