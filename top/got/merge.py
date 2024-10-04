from typing import List
from tqdm import tqdm
import ast

from top.got import PROMPTS_PATH
import os


def get_merge_prompt_got(
    sentences: List[str],
    inputs: List[List[str]],
    outputs: List[List[str]],
    description: str,
) -> List[str]:
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
    if len(sentences) == 0:
        return []
    prompts = []
    merge_prompt = open(
        os.path.join(PROMPTS_PATH, f"merge/{description}.txt"), "r"
    ).read()
    if "intersection" in description and len(inputs[0]) == 4:
        merge_prompt = open(
            os.path.join(PROMPTS_PATH, f"merge/{description}_4.txt"), "r"
        ).read()
    for i in tqdm(range(len(sentences))):
        prompt = merge_prompt
        K = []
        for j, (_, c_out) in enumerate(zip(inputs[i], outputs[i])):
            c_out = c_out.strip()
            # Answer extraction
            for trigger in ["Output:", "Merged list:", "Combined Output:"]:
                if trigger in c_out:
                    c_out = c_out[c_out.find(trigger) + len(trigger) :].strip()
            c_out = c_out.split("\n\n")[0].strip()
            if "keyword" not in description:
                c_out = c_out.split("\n")[0].strip()
            if "set_intersection" in description:
                if any([col in c_out.lower() for col in ["none", "no intersection"]]):
                    c_out = "[]"
            K.append(c_out)
        if "sorting" in description:
            length = len(ast.literal_eval(K[0]))
            prompt = prompt.format(
                length=length,
                length_combined=2 * length,
                input_list1=K[0],
                input_list2=K[1],
            )
        elif "keyword_counting" in description:
            if len(K) >= 2:
                prompt = prompt.format(dictionary_1=K[0], dictionary_2=K[1])
            else:
                prompt = prompt.format(dictionary_1=K[0], dictionary_2={})
        elif "intersection" in description:
            K = [element.strip() for element in K]
            K = [
                f"[{element[2:].strip()}" if element.startswith("[,") else element
                for element in K
            ]
            try:
                for element in K:
                    ast.literal_eval(element)
            except Exception as e:
                raise ValueError(f"Error {e}: The elements of {K} are not all lists.")
            if len(K) == 2:
                prompt = prompt.format(input1=K[0], input2=K[1])
            else:
                prompt = prompt.format(
                    input1=K[0], input2=K[1], input3=K[2], input4=K[3]
                )
        else:
            raise ValueError(
                f"Description '{description}' is not supported. Check `get_merge_prompt_got`."
            )
        if i == 0:
            print(f"{i}===\n{prompt}\n===")
        prompts.append(prompt)
    return prompts
