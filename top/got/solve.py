from top.got import PROMPTS_PATH
import os


def get_solve_prompt_got(sentence: str, problem_name: str, description: str):
    """
    Get the solve prompt for each of the GoT tasks.
    - sentence : input for which we want the output.
    - problem_name: GoT task of interest.
    - description: standard or cot
    """
    solve_prompt = (
        open(os.path.join(PROMPTS_PATH, f"{description}/{problem_name}.txt"), "r")
        .read()
        .strip()
    )
    if problem_name == "sorting":
        prompt = solve_prompt.strip().format(input=sentence)
    elif problem_name == "set_intersection":
        set1, set2 = sentence.strip().split(" + ")
        prompt = solve_prompt.strip().format(set1=set1, set2=set2)
    else:
        prompt = solve_prompt.strip().format(input_text=sentence)
    return prompt
