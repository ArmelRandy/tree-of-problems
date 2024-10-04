from typing import List
from sentence_splitter import SentenceSplitter
from top.bbh.utils import breakdown, z

splitter = SentenceSplitter(language="en")


def divide_bbh(prompts: List[str], n_splits: int, description: str) -> List[List[str]]:
    """
    """
    list_of_subproblems = []
    for prompt in prompts:
        if description == "boolean_expressions":
            input = prompt.replace("is", "").strip()
            if input.startswith("(") and input.endswith(")"):
                # print(f"{input} starts with a '(' but does not end with ')'.")
                _, intervals, _ = z(input)
                if intervals[0] == len(input) - 1:
                    input = input[1:-1].strip()
            _, intervals, roots = z(input)
            exp_to_code = {}
            expression = f"{input}"
            for j, root in enumerate(roots):
                exp = input[root : intervals[root] + 1]
                exp_to_code[chr(65 + j)] = exp
                expression = expression.replace(exp, chr(65 + j), 1)
            trigger = "or"
            problems = expression.split(trigger)
            if len(problems) == 1:
                trigger = "and"
                problems = expression.split(trigger)
            if len(problems) == 1:
                problems = [problems[0]]
            elif len(problems) == 2:
                problems = [problems[0], problems[1]]
            else:
                problems = [
                    f"{trigger}".join(problems[0:2]),
                    f"{trigger}".join(problems[2:]),
                ]
            subproblems = []
            for problem in problems:
                for k, v in exp_to_code.items():
                    problem = problem.replace(k, v, 1)
                subproblems.append(f"{problem.strip()} is")
        elif description == "dyck_languages":
            trigger = "Input: "
            braces = prompt[prompt.find(trigger) + len(trigger) :].strip()
            braces = braces.split(" ")
            problems = breakdown(braces, n_splits)
            problems = [" ".join(problem) for problem in problems]
            subproblems = []
            for j, problem in enumerate(problems):
                subproblem = prompt[: prompt.find(trigger) + len(trigger)]
                if j == 0:
                    subproblem += problem
                else:
                    subproblem += (
                        f"<subproblem>{' '.join(problems[:j])}</subproblem> {problem}"
                    )
                subproblems.append(subproblem)
        elif description == "hyperbaton":
            sentences = prompt.split("\n")
            common = sentences[0].strip()
            steps = sentences[-1].strip().split(" ")
            last = steps[-1]
            steps = steps[:-1]
            subproblems = breakdown(steps, n_splits)
            subproblems = [
                f"{common}\n{' '.join(subproblem + [last])}"
                for subproblem in subproblems
            ]
        elif description == "multistep_arithmetic_two":
            trigger = None
            for candidate in [") + (", ") - (", ") * (", ") / ("]:
                if candidate in prompt:
                    trigger = candidate
                    break
            if trigger:
                left, right = prompt.split(trigger)
                left = left.replace("((", "")
                right = right.replace("))", "")
                subproblems = [left.strip() + " =", right]
            else:
                # split `x1 op1 x1 op2 x3 op3 x4 =`
                exp = prompt.replace(" ", "").replace("=", "")
                numbers = []
                ops = []
                i = 0
                while i < len(exp):
                    if exp[i] == "-":
                        j = i + 1
                    else:
                        j = i
                    while (j < len(exp) and exp[j].isdigit()) or (
                        j < len(exp) and exp[j] == "*"
                    ):
                        if exp[j : j + 2] == "*-":
                            j += 2
                        else:
                            j += 1
                    numbers.append(exp[i:j])
                    try:
                        ops.append(exp[j])
                    except:
                        pass
                    i = j + 1
                if len(numbers) == 1:
                    idx = [i for i, c in enumerate(numbers[0]) if c == "*"][1]
                    subproblems = [f"{numbers[0][:idx]} =", f"{numbers[0][idx+1:]} ="]
                elif len(numbers) == 2:
                    subproblems = numbers
                elif len(numbers) == 3:
                    if "*" in numbers[1]:
                        subproblems = [
                            f"{numbers[0]} {ops[-1]} {numbers[-1]} =",
                            f"{numbers[1]} =",
                        ]
                    elif "*" in numbers[0]:
                        subproblems = [
                            f"{numbers[0]} =",
                            f"{numbers[1]} {ops[-1]} {numbers[-1]} =",
                        ]
                    else:
                        subproblems = [
                            f"{numbers[0]} {ops[0]} {numbers[1]} =",
                            f"{numbers[-1]} =",
                        ]
                else:  # len(numbers) == 4
                    if ops[1] == "+":
                        subproblems = [
                            f"{numbers[0]} {ops[0]} {numbers[1]} =",
                            f"{numbers[2]} {ops[-1]} {numbers[3]} =",
                        ]
                    else:
                        subproblems = [
                            f"{numbers[0]} {ops[0]} {numbers[1]} {ops[1]} {numbers[2]} =",
                            f"{numbers[-1]} =",
                        ]
        elif description == "navigate":
            sentences = splitter.split(prompt)
            common = sentences[0]
            steps = sentences[1:]
            steps = breakdown(steps, n_splits)
            subproblems = [f"{common} {' '.join(step)}" for step in steps]
            subproblems = subproblems + [f"{common} Always face forward."] * (
                n_splits - len(subproblems)
            )
        elif "object_counting" in description:
            sentence, question = splitter.split(prompt)
            steps = sentence.split(", ")
            steps = [
                step.replace("I have ", "").replace("and ", "").replace(".", "").strip()
                for step in steps
            ]
            subproblems = []
            list_of_steps = breakdown(steps, n_splits)
            for element in list_of_steps:
                prompt = "I have "
                if len(element) == 1:
                    prompt += f"{element[0]}."
                else:
                    for item in element[:-1]:
                        prompt += f"{item}, "
                    prompt += f"and {element[-1]}."
                prompt += f" {question}"
                subproblems.append(prompt)
        elif "tracking_shuffled_objects" in description:
            sentences = splitter.split(prompt)
            # Find the initial matching person: item
            temp = sentences[1][
                sentences[1].find(":") + 1 : sentences[1].find(".")
            ].strip()
            temps = temp.split(", ")
            trigger = None
            matching = {}
            for candidate in [" is playing ", " has a ", " is dancing with ", " gets "]:
                if candidate in temps[0]:
                    trigger = candidate
                    break
            for temp in temps:
                a, b = temp.split(trigger)
                a, b = a.strip(), b.strip()
                matching[a] = b

            common = sentences[:3]
            steps = sentences[3:-1]
            question = sentences[-1]
            steps = breakdown(steps, n_splits)
            subproblems = []
            for i, step in enumerate(steps):
                subproblem = " ".join(common)
                for j, elt in enumerate(step):
                    if j == 0:
                        elt = elt.replace("Then, ", "First, ")
                        # elt = elt.replace("Finally, ", "First, ")
                    elif j < len(step) - 1:
                        pass
                    else:
                        elt = elt.replace("Then, ", "Finally, ")
                    subproblem += f" {elt}"
                if i > 0:
                    # Only the first subproblem starts with a known matching
                    # That is the matching of the big problem.
                    for k, (a, b) in enumerate(matching.items()):
                        subproblem = subproblem.replace(
                            f"{a}{trigger}{b}", f"{a}{trigger}<matching {k}>"
                        )
                subproblem += f" {question}"
                subproblems.append(subproblem)
        elif description == "word_sorting":
            header = "Sort the following words alphabetically: List:"
            steps = prompt[len(header) :].strip().split(" ")
            steps = breakdown(steps, n_splits)
            subproblems = []
            for step in steps:
                subproblems.append(f"{header} {' '.join(step)}")
        elif description == "web_of_lies":
            sentences = splitter.split(prompt)
            common = sentences[0]
            first_sentence = sentences[1]
            question = sentences[-1]
            steps = sentences[2:-1]

            names = []
            for sentence in steps:
                names.append(sentence.split()[0])

            subproblems = []
            steps = breakdown(steps, n_splits)
            names = breakdown(names, n_splits)
            for i, step in enumerate(steps):
                subproblem = " ".join(step) + f" Does {names[i][-1]} tells the truth?"
                if i > 0:
                    subproblem = f"{names[i-1][-1]} <truth or lie>. " + subproblem
                else:
                    subproblem = first_sentence + " " + subproblem
                subproblem = common + " " + subproblem
                subproblems.append(subproblem)
        else:
            raise ValueError(f"The description {description} is not supported!")
        list_of_subproblems.append(subproblems)

    return list_of_subproblems
