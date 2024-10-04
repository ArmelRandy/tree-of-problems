from typing import List
from top.bbh.utils import breakdown
from sentence_splitter import SentenceSplitter

splitter = SentenceSplitter(language="en")


def divide_got(prompts: List[str], n_splits: int, description: str) -> List[List[str]]:
    list_of_subproblems = []
    for prompt in prompts:
        if "sorting" in description:
            L = prompt.strip().strip("][").split(", ")
            L = [int(element) for element in L]
            left, right = L[0 : len(L) // 2], L[len(L) // 2 :]
            list_of_subproblems.append([str(left), str(right)])
        elif "keyword" in description:
            steps = splitter.split(prompt)
            steps = breakdown(steps, n_splits)
            steps = [" ".join(step) for step in steps]
            list_of_subproblems.append(steps)
        elif "intersection" in description:
            set1, set2 = prompt.split(" + ")
            set1 = set1.strip().strip("][").split(", ")
            set2 = set2.strip().strip("][").split(", ")
            set1 = [int(element) for element in set1]
            set2 = [int(element) for element in set2]
            a, b = set1[: len(set1) // 2], set1[len(set1) // 2 :]
            c, d = set2[: len(set2) // 2], set2[len(set2) // 2 :]
            if n_splits == 4:
                list_of_subproblems.append(
                    [
                        f"{str(a)} + {str(c)}",
                        f"{str(a)} + {str(d)}",
                        f"{str(b)} + {str(c)}",
                        f"{str(b)} + {str(d)}",
                    ]
                )
            else:  # n_splits = 2
                if len(set1) > len(set2):
                    list_of_subproblems.append(
                        [f"{str(a)} + {str(set2)}", f"{str(b)} + {str(set2)}"]
                    )
                else:
                    list_of_subproblems.append(
                        [f"{str(set1)} + {str(c)}", f"{str(set1)} + {str(d)}"]
                    )
        else:
            raise ValueError(
                f"Description {description} is not supported. Check `divide_got`!"
            )
    return list_of_subproblems


from top.got import PROMPTS_PATH
import os
import ast
from top.bbh.utils import breakdown

# Divide by using the LLM
def divide_got_(
    prompts: List[str],
    n_splits: int,
    description: str,
    generate_fn,  # should return a List[List[str]] when given a List[str]
) -> List[List[str]]:
    trigger = "Output:"
    list_of_subproblems = []
    for prompt in prompts:
        if "sorting" in description:
            prompt = (
                open(os.path.join(PROMPTS_PATH, "divide/sorting.txt"), "r")
                .read()
                .strip()
                .format(input=prompt)
            )
            output = generate_fn([prompt])[0][0]
            if trigger in output:
                output[output.find(trigger) + len(trigger) :].strip()
            output = ast.literal_eval(output)
            list_of_subproblems.append([str(output["List 1"]), str(output["List 2"])])
        elif "keyword" in description:
            """
            prompt = open(os.path.join(PROMPTS_PATH, "divide/keyword_counting.txt"), "r").read().strip().format(input_text=prompt)
            output = generate_fn([prompt])[0][0]
            if trigger in output:
                output[output.find(trigger) + len(trigger):].strip()
            output = ast.literal_eval(output)
            list_of_subproblems.append(
                [
                    output["Paragraph 1"],
                    output["Paragraph 2"],
                    output["Paragraph 3"],
                    output["Paragraph 4"]
                ]
            )
            """
            prompt = (
                open(
                    os.path.join(PROMPTS_PATH, "divide/keyword_counting_sentence.txt"),
                    "r",
                )
                .read()
                .strip()
                .format(input_text=prompt)
            )
            output = generate_fn([prompt])[0][0]
            if trigger in output:
                output[output.find(trigger) + len(trigger) :].strip()
            output = ast.literal_eval(output)
            problems = [output[f"Sentence {j+1}"] for j in range(len(output))]
            problems = breakdown(problems, n_splits=n_splits)
            subproblems = [" ".join(element) for element in problems]
            list_of_subproblems.append(subproblems)
        elif "set_intersection" in description:
            set1, set2 = prompt.split(" + ")
            s1 = ast.literal_eval(set1)
            s2 = ast.literal_eval(set2)
            if len(s1) < len(s2):
                set1, set2 = set2, set1
            prompt1 = (
                open(os.path.join(PROMPTS_PATH, "divide/set_intersection.txt"), "r")
                .read()
                .strip()
                .format(input=set1)
            )
            output = generate_fn([prompt1])[0][0]
            if trigger in output:
                output[output.find(trigger) + len(trigger) :].strip()
            output = ast.literal_eval(output)
            a, b = output["List 1"], output["List 2"]
            if n_splits == 4:
                prompt2 = (
                    open(os.path.join(PROMPTS_PATH, "divide/set_intersection.txt"), "r")
                    .read()
                    .strip()
                    .format(input=set2)
                )
                output = generate_fn([prompt2])[0][0]
                if trigger in output:
                    output[output.find(trigger) + len(trigger) :].strip()
                output = ast.literal_eval(output)
                c, d = output["List 1"], output["List 2"]
                list_of_subproblems.append(
                    [
                        f"{str(a)} + {str(c)}",
                        f"{str(a)} + {str(d)}",
                        f"{str(b)} + {str(c)}",
                        f"{str(b)} + {str(d)}",
                    ]
                )
            else:
                list_of_subproblems.append(
                    [f"{str(a)} + {str(set2)}", f"{str(b)} + {str(set2)}"]
                )
        else:
            raise ValueError(
                f"Description {description} is not supported. Check `divide_got_`!"
            )
    return list_of_subproblems
