import re
import torch
from typing import List
from transformers import StoppingCriteria
from torch.utils.data import IterableDataset


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        done = []
        for decoded_generation in decoded_generations:
            done.append(
                any(
                    [
                        stop_string in decoded_generation
                        for stop_string in self.eof_strings
                    ]
                )
            )
        return all(done)


class TokenizedDataset(IterableDataset):
    """Tokenize and preprocess the dataset, where the dataset is a list of instructions (str)"""

    def __init__(self, tokenizer, dataset):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.outputs = self.tokenizer(self.dataset, padding=True, return_tensors="pt")

    def __iter__(self):
        for i in range(len(self.dataset)):
            yield {
                "input_ids": self.outputs.input_ids[i],
                "attention_mask": self.outputs.attention_mask[i],
                "index_prompt": torch.tensor(i, dtype=torch.int32),
            }


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


# https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/base.py#L83
def _stop_at_stop_token(decoded_string, stop_tokens):
    """
    Produces the prefix of decoded_string that ends at the first occurrence of
    a stop_token.
    WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
    itself.
    """
    min_stop_index = len(decoded_string)
    for stop_token in stop_tokens:
        stop_index = decoded_string.find(stop_token)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return decoded_string[:min_stop_index].rstrip()


from top.bbh.utils import BBH_TASKS


def answer_extraction(task: str, prompt: str):
    if task in ["algebraic", "coin"] + BBH_TASKS:
        trigger = "So the answer is"
        if trigger in prompt:
            return prompt[prompt.find(trigger) + len(trigger) :].strip()
        else:
            return prompt
    elif task == "concatenation":
        trigger = ' outputs "'
        if trigger in prompt:
            return prompt[
                prompt.find(trigger) + len(trigger) : prompt.rfind('"')
            ].strip()
        else:
            return prompt
    elif task in ["keyword_counting", "set_intersection", "sorting"]:
        trigger = "Output:"
        if trigger in prompt:
            return prompt[prompt.find(trigger) + len(trigger) :].strip()
        else:
            return prompt
    else:
        raise ValueError(f"Unsupported task '{task}'")


def get_best_sentence(sentences: List[str], problem_name: str, verbose=True):
    """
    Takes as input a list of sentences (answers) and compute a score for each of them and return the sentence with the highest score
    Arguments
    ---------
        - sentences : List[str],
            sentences we would like to score
    """
    if len(sentences) == 1:
        return sentences[0]

    answers = []
    occ = {}
    for i in range(len(sentences)):
        answer = answer_extraction(task=problem_name, prompt=sentences[i])
        answers.append(answer)
        occ[answer] = occ.get(answer, 0) + 1
    max_key, max_occ = None, None
    for key in occ:
        if max_key is None or occ[key] > max_occ:
            max_key, max_occ = key, occ[key]
    idx = 0
    while idx < len(sentences):
        if answers[idx] == max_key:
            break
        idx += 1

    if verbose:
        prompt = "===\n"
        prompt = "The most consistent answer between the following\n"
        for i in range(len(sentences)):
            prompt += f"{i+1}. {sentences[i]}\n"
        prompt += f"Is\n{sentences[idx]}"
        prompt += "\n==="
        print(prompt)

    return sentences[idx]
