from typing import List
from tqdm import tqdm
from top.concatenation import PROMPTS_PATH
import os

merge_prompt = "We would like to get the concatenation of the last letters of a large list of words. For this purpose, you are provided with the concatenation of the last letters of multiple sublists of the large list. Take inspiration of those value to compute the concatenation of the the last letters of the large list.\n\n"


def get_merge_prompt(
    sentences: List[str], inputs: List[List[str]], outputs: List[List[str]]
):
    """
    Build a few-shot prompt for merging subproblems solutions into a unique solutions.
    Arguments
    ---------
        -  sentences: List[str],
            Problems to be solved.
        - inputs : List[List[str]],
            List of subproblems' inputs.
        - outputs : List[List[str]],
            List of subproblems' outputs.
    Returns
    -------
        - List[str] :
            Prompt for solving the problem defined by each sentence by taking inspiration from the subproblems solutions.
    """
    prompts = []
    for i in tqdm(range(len(sentences))):
        prompt = (
            merge_prompt
            + open(os.path.join(PROMPTS_PATH, "merge/merge.txt"), "r").read().strip()
            + "\n\n"
        )
        for j in range(len(inputs[i])):
            prompt += f'Q: "{inputs[i][j]}"\nA: {outputs[i][j]}\n\n'
        prompt += "We can conclude that\n\n"
        prompt += f'Q: "{sentences[i]}"\nA:'
        prompts.append(prompt)
    return prompts


def get_merge_prompt_l2m(
    sentences: List[str], inputs: List[List[str]], outputs: List[List[str]]
):
    prompts = []
    for i in tqdm(range(len(sentences))):
        prompt = (
            merge_prompt
            + open(os.path.join(PROMPTS_PATH, "merge/l2m.txt"), "r").read().strip()
            + "\n\n"
        )
        for j in range(len(inputs[i])):
            prompt += f"Q: {inputs[i][j]}\nA: {outputs[i][j]}\n\n"
        prompt += f"Q: {sentences[i]}\nA:"
        prompts.append(prompt)
    return prompts
