from typing import List
from tqdm import tqdm

from top.coin import PROMPTS_PATH
import os

merge_prompt = "Answer to the following questions by yes or no.\n\n"


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
        if len(inputs[i]) >= 2:
            prompt = (
                merge_prompt
                + open(os.path.join(PROMPTS_PATH, "merge/merge_2.txt"), "r")
                .read()
                .strip()
                + "\n\n"
            )
            for j in range(len(inputs[i])):
                prompt += f"Q: {inputs[i][j]}\nA: {outputs[i][j]}\n\n"
            prompt += "We can conclude that\n\n"
            prompt += f"Q: {sentences[i]}\nA:"
        else:
            prompt = (
                merge_prompt
                + open(os.path.join(PROMPTS_PATH, "merge/merge.txt"), "r")
                .read()
                .strip()
                + "\n\n"
            )
            prompt += f"Q: {sentences[i]}\nA:"
            output = outputs[i][0]
            if "So the answer is " in output:
                answer = output[
                    output.find("So the answer is ")
                    + len("So the answer is ") : output.rfind(".")
                ].strip()
            else:
                answer = output
            assert answer in [
                "yes",
                "no",
            ], "The answer for the coin problem should be either 'yes' or 'no'."
            if "started heads up" in output:
                prompt = prompt.replace(
                    "<status>", "heads" * (answer == "yes") + "tails" * (answer == "no")
                )
            elif "started tails up" in output:
                prompt = prompt.replace(
                    "<status>", "tails" * (answer == "yes") + "heads" * (answer == "no")
                )
            else:
                raise ValueError(
                    f"{output} contains neither 'started heads up' nor 'started tails up'."
                )
            # print(f"===\nQ: {inputs[i][0]}\nA: {output}\n\n{prompt}\n===")
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
