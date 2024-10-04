from typing import List
from tqdm import tqdm
import os

from top.bbh import PROMPTS_PATH
from top.bbh.utils import PROBLEM_TO_FILENAME


def get_merge_prompt_bbh(
    sentences: List[str],
    inputs: List[List[str]],
    outputs: List[List[str]],
    description: str,
):
    """
    Build a few-shot prompt for merging subproblems solutions into a unique solution.
    Arguments
    ---------
        -  sentences: List[str],
            Problems to be solved.
        - inputs : List[List[str]],
            List of subproblems' inputs.
        - outputs : List[List[str]],
            List of subproblems' outputs.
        - description: str
            Name of the task of interest in BBH
    Returns
    -------
        - List[str] :
            Prompt for solving the problem defined by each sentence by taking inspiration from the subproblems solutions.
    """
    merge_prompt = (
        open(
            os.path.join(PROMPTS_PATH, f"merge/{PROBLEM_TO_FILENAME[description]}.txt"),
            "r",
        )
        .read()
        .strip()
        + "\n\n"
    )

    prompts = []
    if any(
        [
            element in description
            for element in [
                "hyperbaton",
                "multistep_arithmetic_two",
                "object_counting",
                "word_sorting",
                "boolean_expressions",
            ]
        ]
    ):
        for i in tqdm(range(len(sentences))):
            if "word_sorting" in description:
                trigger = "So the answer is "
                outputs[i] = [
                    (
                        element[
                            element.find(trigger) + len(trigger) : element.rfind(".")
                        ]
                        if element.find(trigger) >= 0
                        else element
                    )
                    for element in outputs[i]
                ]
                prompt = ""
                for j in range(len(inputs[i])):
                    prompt += (
                        f"- Premise {j+1}\n\nQ: {inputs[i][j]}\nA: {outputs[i][j]}\n\n"
                    )
                prompt += "We can conclude that\n\n"
                prompt += f"Q: {sentences[i]}\nA:"
                prompt += " Let's think step by step."
                prompt += f'\nAccording to the "Premise 1", we have A: {outputs[i][0]}. We then build a list A = {outputs[i][0]}.'
                try:
                    prompt += f'\nAccording to the "Premise 2", we have A: {outputs[i][1]}. We then build a list B = {outputs[i][1]}.'
                    prompt += "\nLet's merge A and B in order to build a list C."
                except Exception as e:
                    pass
            elif "object_counting" in description:
                L = len(outputs[i])
                prompt = ""
                for j in range(len(inputs[i])):
                    prompt += f"Q: {inputs[i][j]}\nA: {outputs[i][j]}\n\n"
                prompt += "We can conclude that\n\n"
                prompt += f"Q: {sentences[i]}\nA: Let's think step by step.\n"
            else:
                prompt = ""
                for j in range(len(inputs[i])):
                    prompt += f"Q: {inputs[i][j]}\nA: {outputs[i][j]}\n\n"
                prompt += "We can conclude that\n\n"
                prompt += f"Q: {sentences[i]}\nA:"
            prompts.append(prompt)
    elif "navigate" in description:
        for i in tqdm(range(len(sentences))):
            prompt = sentences[i]
            output = outputs[i][0]
            if "So the answer is " in output:
                end = output.find("So the answer is")
                start = output.rfind(":")
                end_position = output[start + 1 : end].strip().replace(".", "?")
            else:
                input = inputs[i][0]
                end_position = input[input.find("(") : input.find("?") + 1]
            prompt = prompt.replace(
                "the point (0, 0), facing the positive y-axis?",
                f"the point {end_position}",
            )
            prompt = f"Q: {prompt}\nA:"
            prompts.append(prompt)
    elif "tracking_shuffled_objects" in description:
        for i in tqdm(range(len(sentences))):
            prompt = sentences[i]
            output = outputs[i][0]
            # Build the matching obtained by solving the previous subproblem
            # And use it as input for the current subproblem
            if "So the answer is " in output:
                output = output[
                    output.find("So the answer is ")
                    + len("So the answer is ") : output.rfind(".")
                ].strip()
            output = output.split(", ")
            output = [element.split(":") for element in output]
            try:
                matching = {
                    element[0].strip(): element[1].strip() for element in output
                }
                count = 0
                for _, v in matching.items():
                    prompt = prompt.replace(f"<matching {count}>", v)
                    count += 1
                prompt = f"Q: {prompt}\nA:"
                prompts.append(prompt)
            except Exception as e:
                print(f"An error occurred: {e} in {output}\n{prompt}")
                prompts.append(f"Q: {prompt}\nA:")
    elif "web_of_lies" in description:
        for i in tqdm(range(len(sentences))):
            prompt = sentences[i]
            output = outputs[i][0]
            if "So the answer is " in output:
                output = output[
                    output.find("So the answer is ")
                    + len("So the answer is ") : output.rfind(".")
                ].strip()
            if output == "Yes":
                prompt = prompt.replace("<truth or lie>", "tells the truth")
            else:
                prompt = prompt.replace("<truth or lie>", "lies")
            prompt = f"Q: {prompt}\nA:"
            print(f"prompt: {prompt}, output: {output}")
            prompts.append(prompt)
    elif "dyck_languages" in description:
        reverse = {"]": "[", ")": "(", "}": "{", ">": "<"}
        for i in tqdm(range(len(sentences))):
            prompt = sentences[i]
            output = outputs[i][0]
            if "So the answer is " in output:
                output = output[
                    output.find("So the answer is ")
                    + len("So the answer is ") : output.rfind(".")
                ]
            output = output.strip()
            if all([c in [")", "}", "]", ">"] for c in output.replace(" ", "")]):
                # Only ")", "}"", "]" or ">"
                begin, end = "<subproblem>", "</subproblem>"
                i_begin, i_end = prompt.find(begin), prompt.find(end)
                if output == "":
                    prompt = prompt.replace(
                        prompt[i_begin : i_end + len(end)] + " ", ""
                    )
                elif len(output.split(" ")) > 50:
                    prompt = prompt.replace(begin, "").replace(end, "")
                else:
                    try:
                        output = " ".join([reverse[c] for c in output.split(" ")[::-1]])
                        prompt = prompt.replace(
                            prompt[i_begin : i_end + len(end)], output
                        )
                    except Exception as e:
                        # Answer with bad format
                        print(f"This error occured: {e}.")
            else:
                begin, end = "<subproblem>", "</subproblem>"
                prompt = prompt.replace(begin, "").replace(end, "")
            prompt = f"Q: {prompt}\nA:"
            prompts.append(prompt)
    else:
        pass
    prompts = [merge_prompt + prompt for prompt in prompts]
    return prompts


if __name__ == "__main__":
    pass
