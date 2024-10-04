from typing import List
from sentence_splitter import SentenceSplitter
from top.coin import PROMPTS_PATH
import os


def get_solve_prompt(prompt: str, description: str, k=8):
    """
    Build the few-shot prompt to solve the problem of interest
    """
    out = "Answer to the following questions by 'yes' or 'no'.\n\n"
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
    Returns the chain of thought of Coin Flip
    """
    splitter = SentenceSplitter(language="en")
    chain_of_thoughts = []
    for sample in samples:
        prompt = ""
        sentences = splitter.split(text=sample)
        names = []
        for sentence in sentences:
            trigger = "flips the coin"
            idx = sentence.find(trigger)
            if idx == -1:
                continue
            name = sentence[0:idx].strip()
            names.append(name)
        if len(names) == 0:
            prompt = "The coin was flipped by no one. So the coin was flipped 0 times. The coin started heads up, and it was not flipped, so it is still heads up. So the answer is yes."
        elif len(names) == 1:
            prompt = f"The coin was flipped by {names[0]}. So the coin was flipped 1 time, which is an odd number. The coin started heads up, so after an odd number of flips, it will be tails up. So the answer is no."
        elif len(names) == 2:
            prompt = f"The coin was flipped by {names[0]} and {names[1]}. So the coin was flipped 2 times, which is an even number. The coin started heads up, so after an even number of flips, it will still be heads up. So the answer is yes."
        else:
            prompt += f"The coin was flipped by {', '.join(names[:-1])} and {names[-1]}. So the coin was flipped {len(names)} times, which is an "
            if len(names) % 2 == 0:
                prompt += "even number. The coin started heads up, so after an even number of flips, it will still be heads up. So the answer is yes."
            else:
                prompt += "odd number. The coin started heads up, so after an odd number of flips, it will be tails up. So the answer is no."
        chain_of_thoughts.append(prompt)
    return chain_of_thoughts
