import re
from datasets import load_dataset
from sentence_splitter import SentenceSplitter

splitter = SentenceSplitter(language="en")

LEFT = {"+y": "-x", "-y": "+x", "+x": "+y", "-x": "-y"}
RIGHT = {"+y": "+x", "-y": "-x", "+x": "-y", "-x": "+y"}
AROUND = {"+y": "-y", "-y": "+y", "+x": "-x", "-x": "+x"}


def extract_number(sentence):
    # Define a regular expression pattern to match numbers
    pattern = r"\d+"  # This pattern matches one or more digits

    # Use the findall method from the re module to extract all matching numbers
    numbers = re.findall(pattern, sentence)

    # If there are numbers found, return the first one (you can adjust this as needed)
    if numbers:
        return int(
            numbers[0]
        )  # Convert the first number found to an integer and return it
    else:
        return None  # Return None if no numbers are found in the sentence


def navigate():
    dataset = load_dataset("lukaemon/bbh", "navigate")
    instructions = []
    outputs = []
    for element in dataset["test"]:
        instruction = element["input"]
        target = element["target"]
        instruction = instruction[
            instruction.find("? ") + 2 : instruction.find("Options:")
        ].strip()
        x = 0
        y = 0
        sentences = splitter.split(text=instruction)
        position = "+y"
        for sentence in sentences:
            if sentence == "Always face forward.":
                # Nothing happens
                pass
            elif "Turn" in sentence:
                if "left" in sentence:
                    position = LEFT[position]
                elif "right" in sentence:
                    position = RIGHT[position]
                elif "around" in sentence:
                    position = AROUND[position]
                else:
                    # Never happening
                    print(instruction)
            elif "Take" in sentence:
                number_of_steps = extract_number(sentence)
                if "forward" in sentence:
                    if position == "+x":
                        x += number_of_steps
                    elif position == "-x":
                        x -= number_of_steps
                    elif position == "+y":
                        y += number_of_steps
                    elif position == "-y":
                        y -= number_of_steps
                    else:
                        pass
                elif "backward" in sentence:
                    if position == "+x":
                        x -= number_of_steps
                    elif position == "-x":
                        x += number_of_steps
                    elif position == "+y":
                        y -= number_of_steps
                    elif position == "-y":
                        y += number_of_steps
                    else:
                        pass
                elif "left" in sentence:
                    if position == "+x":
                        y += number_of_steps
                    elif position == "-x":
                        y -= number_of_steps
                    elif position == "+y":
                        x -= number_of_steps
                    elif position == "-y":
                        x += number_of_steps
                    else:
                        print(f"Not happening 1 {position}")
                elif "right" in sentence:
                    if position == "+x":
                        y -= number_of_steps
                    elif position == "-x":
                        y += number_of_steps
                    elif position == "+y":
                        x += number_of_steps
                    elif position == "-y":
                        x -= number_of_steps
                    else:
                        print(f"Not happening 2 {position}")
                else:
                    # same direction
                    if position == "+x":
                        x += number_of_steps
                    elif position == "-x":
                        x -= number_of_steps
                    elif position == "+y":
                        y += number_of_steps
                    elif position == "-y":
                        y -= number_of_steps
                    else:
                        print(f"Not happening 3 {position}")
                    # print(instruction)
            else:
                # Never happening
                print(instruction)
        # if target != "Yes":
        #    print(f"({x}, {y}), target = {target}")
        instructions.append(
            f"If you follow these instructions, what are the coordinates of the end point if you start at the point (0, 0), facing the positive y-axis? {instruction}"
        )
        outputs.append(f"({x}, {y})")
    return [(a, b) for (a, b) in zip(instructions, outputs)]


def temporal():
    dataset = load_dataset("lukaemon/bbh", "temporal_sequences")
    pairs = []
    for i, example in enumerate(dataset["test"]):
        instruction = example["input"]
        target = example["target"]
        idx = instruction.find("Options:")
        instruction, leftover = (
            instruction[:idx].strip(),
            instruction[idx + len("Options:") :].strip(),
        )
        target = leftover[leftover.find(target) + len(target) :].strip()
        target = target.split("\n")[0]
        pairs.append((instruction, target))
    return pairs


def tracking_shuffled_objects(description="seven"):
    dataset = load_dataset(
        "lukaemon/bbh", f"tracking_shuffled_objects_{description}_objects"
    )
    pairs = []
    for i, example in enumerate(dataset["test"]):
        instruction = example["input"]
        sentences = splitter.split(instruction)
        names = sentences[0].split(", ")
        last = names.pop()
        last = last.strip()
        if last.startswith("and"):
            last = last[3:]
        idx = last.find("are")
        final_name = last[:idx].strip()
        last = last[idx + 3 :].strip()
        names.append(final_name)
        matching = {}
        if "dancers" in last:
            sentence = sentences[1][
                sentences[1].find("partner:") + len("partner:") :
            ].strip()
            trigger = "is dancing with"
            end_point = "what is the assignment of partners?"
        elif "game" in last:
            sentence = sentences[1][
                sentences[1].find("holding a ball:") + len("holding a ball:") :
            ].strip()
            trigger = "has a"
            end_point = "what is the assignment of balls?"
        elif "trade books" in last:
            sentence = sentences[1][
                sentences[1].find("one new book:") + len("one new book:") :
            ].strip()
            trigger = "gets"
            end_point = "what is the assignment of books?"
        elif "soccer match" in last:
            sentence = sentences[1][
                sentences[1].find("to a position:") + len("to a position:") :
            ].strip()
            trigger = "is playing"
            end_point = "what is the assignment of positions?"
        elif "gift exchange" in last:
            sentence = sentences[1][
                sentences[1].find("different color:") + len("different color:") :
            ].strip()
            trigger = "has a"
            end_point = "what is the assignment of gifts?"
        else:
            # Not happening
            print(last)
        # Building the initial matching
        for element in sentence.split(","):
            idx = element.find(trigger)
            a = element[:idx].strip()
            b = element[idx + len(trigger) :].strip()
            a = a.split(" ")[-1].strip()
            b = b[:-1].strip() if b.endswith(".") else b.strip()
            matching[a] = b
            mouvements = []
        for sentence in sentences[2:]:
            if any([sentence.startswith(con) for con in ["First", "Then", "Finally"]]):
                mouvements.append(sentence)
        # Doing the shuffling
        for sentence in mouvements:
            sentence = sentence.split(", ")[-1][:-1].strip()
            trigger = "and"
            left = sentence[: sentence.find(trigger)]
            right = sentence[sentence.find(trigger) + len(trigger) :]
            left = left.strip().split(" ")[0].strip()
            right = right.strip().split(" ")[0].strip()
            matching[left], matching[right] = matching[right], matching[left]
        O = []
        for elt in sentences:
            if elt.startswith("At the end"):
                break
            O.append(elt)
        O.append(elt.split(", ")[0] + ", " + end_point)
        instruction = " ".join(O)
        target = ", ".join([f"{k}: {v}" for k, v in matching.items()])
        # print(f"Q: {instruction}\nA: {target}")
        # print("\n###\n")
        pairs.append((instruction, target))
    return pairs


import itertools


def handle(
    sentences,
    ranking,
    trigger_0=" is the leftmost.",
    trigger_1=" is the second from the left.",
    trigger_2=" is the third from the left.",
    trigger_3=" is the fourth from the left.",
    trigger_4=" is the third from the right.",
    trigger_5=" is the second from the right.",
    trigger_6=" is the rightmost.",
    smaller=" is to the left of ",
    bigger=" is to the right of",
):
    tuples = []
    count = 0
    triggers = [
        trigger_0,
        trigger_1,
        trigger_2,
        trigger_3,
        trigger_4,
        trigger_5,
        trigger_6,
    ]
    triggers = [trigger for trigger in triggers if trigger is not None]
    for sentence in sentences:
        for i, trigger in enumerate(triggers):
            if sentence.endswith(trigger):
                name = sentence[: sentence.find(trigger)].replace("The ", "").strip()
                ranking[name] = i
                count += 1
        if smaller in sentence:
            a, b = sentence.split(smaller)
            a, b = a.replace("The ", "").strip(), b[:-1].replace("the ", "").strip()
            # print(f"{a} < {b}")
            tuples.append((a, b))
            count += 1
        elif bigger in sentence:
            a, b = sentence.split(bigger)
            a, b = a.replace("The ", "").strip(), b[:-1].replace("the ", "").strip()
            # print(f"{a} > {b}")
            tuples.append((b, a))
            count += 1
        else:
            # ignore
            pass
    # print(f"count: {count}.")
    keys_to_complete = [k for (k, v) in ranking.items() if v == -1]
    completed_keys = [v for (_, v) in ranking.items() if v != -1]
    candidates = [i for i in range(len(triggers)) if i not in completed_keys]
    for element in itertools.permutations(candidates):
        temp = {}
        for a, b in zip(keys_to_complete, element):
            temp[a] = b

        if all(
            [temp.get(a, ranking[a]) < temp.get(b, ranking[b]) for (a, b) in tuples]
        ):
            # print(f"MATCH: {temp}\nTuples: {tuples}. Dic: {[(a1, ranking[a1]) for a1 in ranking if ranking[a1] >= 0]}")
            for k in keys_to_complete:
                ranking[k] = temp[k]
            break
    return ranking


def logical_deduction(description):
    dataset = load_dataset("lukaemon/bbh", f"logical_deduction_{description}_objects")
    pairs = []
    for _, example in enumerate(dataset["test"]):
        instruction = example["input"]
        target = example["target"]
        instruction = instruction[: instruction.find("Options:")].strip()
        sentences = splitter.split(text=instruction)
        sentence = (
            ", ".join(sentences[2].split(", ")[1:])
            if not sentences[2].startswith("A fruit stand")
            else sentences[2]
        )
        # print(instruction)
        # print(sentences[2])
        # print(sentence)
        if len(sentence.split(":")) == 2:
            _, right = sentence.split(":")
        else:
            right = sentence
        # remove the full stop
        right = right[:-1]
        names = right.split(", ")
        names = [name.replace("and ", " ").strip() for name in names]
        ranking = {
            name.replace("a ", "").replace("an ", "").strip(): -1 for name in names
        }
        # print(ranking)
        if "birds" in instruction or "books" in instruction:
            if description == "seven":
                trigger_0 = " is the leftmost."
                trigger_1 = " is the second from the left."
                trigger_2 = " is the third from the left."
                trigger_3 = " is the fourth from the left."
                trigger_4 = " is the third from the right."
                trigger_5 = " is the second from the right."
                trigger_6 = " is the rightmost."
            elif description == "five":
                trigger_0 = " is the leftmost."
                trigger_1 = " is the second from the left."
                trigger_2 = " is the third from the left."
                trigger_3 = " is the second from the right."
                trigger_4 = " is the rightmost."
                trigger_5, trigger_6 = None, None
            elif description == "three":
                trigger_0 = " is the leftmost."
                trigger_1 = " is the second from the left."
                trigger_2 = " is the rightmost."
                trigger_3, trigger_4, trigger_5, trigger_6 = None, None, None, None
            else:
                # Not happening
                pass
            smaller = " is to the left of "
            bigger = " is to the right of"
            if "birds" in instruction:
                instruction = f"{instruction} What is the ordering (from left to right) of the birds?"
            else:
                instruction = f"{instruction} What is the ordering (from left to right) of the books?"
        elif "golf tournament" in instruction:
            if description == "seven":
                trigger_0 = " finished first."
                trigger_1 = " finished second."
                trigger_2 = " finished third."
                trigger_3 = " finished fourth."
                trigger_4 = " finished third-to-last."
                trigger_5 = " finished second-to-last."
                trigger_6 = " finished last."
            elif description == "five":
                trigger_0 = " finished first."
                trigger_1 = " finished second."
                trigger_2 = " finished third."
                trigger_3 = " finished second-to-last."
                trigger_4 = " finished last."
                trigger_5, trigger_6 = None, None
            elif description == "three":
                trigger_0 = " finished first."
                trigger_1 = " finished second."
                trigger_2 = " finished last."
                trigger_3, trigger_4, trigger_5, trigger_6 = None, None, None, None
            else:
                # Not happening
                pass
            smaller = " finished above "  # better, smaller rank
            bigger = " finished below "
            instruction = f"{instruction} What is the ordering of the golfers?"
        elif "fruit stand" in instruction:
            if description == "seven":
                trigger_0 = " are the cheapest."
                trigger_1 = " are the second-cheapest."
                trigger_2 = " are the third-cheapest."
                trigger_3 = " are the fourth-most expensive."
                trigger_4 = " are the third-most expensive."
                trigger_5 = " are the second-most expensive."
                trigger_6 = " are the most expensive."
            elif description == "five":
                trigger_0 = " are the cheapest."
                trigger_1 = " are the second-cheapest."
                trigger_2 = " are the third-cheapest."
                trigger_3 = " are the second-most expensive."
                trigger_4 = " are the most expensive."
                trigger_5, trigger_6 = None, None
            elif description == "three":
                trigger_0 = " are the cheapest."
                trigger_1 = " are the second-most expensive."
                trigger_2 = " are the most expensive."
                trigger_3, trigger_4, trigger_5, trigger_6 = None, None, None, None
            else:
                # Not happening
                pass
            smaller = " are less expensive than "
            bigger = " are more expensive than "
            instruction = f"{instruction} What is the ordering (from the cheapest to the most expensive) of the fruits?"
        elif "antique car" in instruction:
            if description == "seven":
                trigger_0 = " is the newest."
                trigger_1 = " is the second-newest."
                trigger_2 = " is the third-newest."
                trigger_3 = " is the fourth-newest."
                trigger_4 = " is the third-oldest."
                trigger_5 = " is the second-oldest."
                trigger_6 = " is the oldest."
            elif description == "five":
                trigger_0 = " is the newest."
                trigger_1 = " is the second-newest."
                trigger_2 = " is the third-newest."
                trigger_3 = " is the second-oldest."
                trigger_4 = " is the oldest."
                trigger_5, trigger_6 = None, None
            elif description == "three":
                trigger_0 = " is the newest."
                trigger_1 = " is the second-newest."
                trigger_2 = " is the oldest."
                trigger_3, trigger_4, trigger_5, trigger_6 = None, None, None, None
            else:
                # Not happening
                pass
            smaller = " is newer than "
            bigger = " is older than"
            instruction = f"{instruction} What is the ordering (from the newest to the oldest) of the vehicles?"
        else:
            # Not happening
            print(instruction)
        ranking = handle(
            sentences,
            ranking,
            trigger_0,
            trigger_1,
            trigger_2,
            trigger_3,
            trigger_4,
            trigger_5,
            trigger_6,
            smaller,
            bigger,
        )
        reverse = {v: k for (k, v) in ranking.items()}
        target = ", ".join([reverse[k].capitalize() for k in sorted(reverse)])
        pairs.append((instruction, target))
    return pairs


def hyperbaton():
    dataset = load_dataset("lukaemon/bbh", "hyperbaton")
    pairs = []
    for example in dataset["test"]:
        instruction = example["input"]
        target = example["target"]
        trigger = "Options:"
        instruction = instruction[instruction.find(trigger) + len(trigger) :].strip()
        sentences = instruction.split("\n")
        sentences = [sentence[3:].strip() for sentence in sentences]
        sentences = [
            f"Answer with Yes or No. Does the following sentence have the correct adjective order?\n{sentence}"
            for sentence in sentences
        ]
        # print(sentences)
        if target == "(A)":
            pairs.append((sentences[0], "Yes"))
            pairs.append((sentences[1], "No"))
        elif target == "(B)":
            pairs.append((sentences[1], "Yes"))
            pairs.append((sentences[0], "No"))
        else:
            pass
    return pairs


def get_dataset(key):
    if key == "boolean_expressions":
        dataset = load_dataset("lukaemon/bbh", "boolean_expressions")
        return [(element["input"], element["target"]) for element in dataset["test"]]
    elif key == "multistep_arithmetic_two":
        dataset = load_dataset("lukaemon/bbh", "multistep_arithmetic_two")
        return [(element["input"], element["target"]) for element in dataset["test"]]
    elif key == "navigate":
        return navigate()
    elif key == "object_counting":
        dataset = load_dataset("lukaemon/bbh", "object_counting")
        return [(element["input"], element["target"]) for element in dataset["test"]]
    elif key == "temporal_sequences":
        return temporal()
    elif "tracking_shuffled_objects" in key:
        if "three" in key:
            description = "three"
        elif "five" in key:
            description = "five"
        else:
            description = "seven"
        return tracking_shuffled_objects(description)
    elif key == "word_sorting":
        dataset = load_dataset("lukaemon/bbh", "word_sorting")
        pairs = [(element["input"], element["target"]) for element in dataset["test"]]
        pairs = [(a, b) for (a, b) in pairs if len(b.split()) <= 16]
        return pairs
    elif "logical_deduction" in key:
        if "three" in key:
            description = "three"
        elif "five" in key:
            description = "five"
        else:
            description = "seven"
        return logical_deduction(description)
    elif key == "hyperbaton":
        return hyperbaton()
    elif key == "web_of_lies":
        dataset = load_dataset("lukaemon/bbh", "web_of_lies")
        return [
            (
                element["input"].replace("Question: ", "Answer with Yes or No.\n"),
                element["target"],
            )
            for element in dataset["test"]
        ]
    elif key == "sports_understanding":
        dataset = load_dataset("lukaemon/bbh", "sports_understanding")
        return [
            (
                element["input"]
                .replace("?", "? Answer with yes or no.")
                .replace(' "', "\n")
                .replace('"', ""),
                element["target"],
            )
            for element in dataset["test"]
        ]
    elif key == "dyck_languages":
        dataset = load_dataset("lukaemon/bbh", "dyck_languages")
        pairs = [(element["input"], element["target"]) for element in dataset["test"]]
        pairs = [
            (a, b)
            for (a, b) in pairs
            if len(a[a.find("Input: ") + len("Input:") :].strip().split(" ")) >= 8
        ]
        return pairs
    else:
        raise ValueError(f"Key {key} is not supported for the dataset BBH.")
