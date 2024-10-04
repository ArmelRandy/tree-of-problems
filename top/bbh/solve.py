from top.bbh import PROMPTS_PATH
from top.bbh.utils import PROBLEM_TO_FILENAME
import os


def get_solve_prompt_bbh(sentence: str, problem_name: str, description: str):
    """
    Get the solve prompt for each of the BBH tasks.
    - sentence : input for which we want the output.
    - problem_name: BBH task of interest.
    - description: standard or cot.
    """
    solve_prompt = (
        open(
            os.path.join(
                PROMPTS_PATH, f"{description}/{PROBLEM_TO_FILENAME[problem_name]}.txt"
            ),
            "r",
        )
        .read()
        .strip()
        + "\n\n"
    )
    return solve_prompt + f"Q: {sentence}\nA: "
