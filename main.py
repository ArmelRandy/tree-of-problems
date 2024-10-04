import os
import json
import argparse
from tqdm import tqdm

from top.models import apply_template
from top.utils import _stop_at_stop_token, get_best_sentence
from top.generator import *

from top.divide import divide
from top.merge import get_merge_prompt as merge_function
from top.solve import get_solve_prompt as solve_function

from typing import List


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Name or path of the model used for text generation.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        help="Name or path of the tokenizer of the model used for text generation",
    )
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        help="Path to the dataframe containing the evaluation data.",
    )
    parser.add_argument(
        "--max_samples", type=int, help="Maximum number of problems to solve."
    )
    parser.add_argument(
        "--temperature", type=float, help="Temperature of the generation."
    )
    parser.add_argument(
        "--top_p", type=float, help="Top_p parameter, for nucleus sampling."
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of output sequences to return for the given prompt. Should be less or equal to `num_beams` in case of beam search.",
    )
    parser.add_argument(
        "--num_beams", type=int, default=1, help="Number of beams, for beam search."
    )
    parser.add_argument("--repetition_penalty", type=float, help="Repetition penalty.")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=75)
    parser.add_argument("--output_dir", type=str, help="Output directory.")
    parser.add_argument("--metadata_dir", type=str, help="Metadata directory.")
    parser.add_argument(
        "--inference_api",
        type=str,
        default="vllm",
        choices=["vllm", "openai", "hf"],
        help="Which API to use for text generation, set to vllm by default.",
    )
    parser.add_argument("--api_key", type=str, default=None, help="OPENAI API KEY.")
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=4,
        help="Batch size for text generation.",
    )
    parser.add_argument(
        "--k", type=int, help="Number of example demonstrations.",
    )
    parser.add_argument("--seed", type=int, help="Seed parameter")
    parser.add_argument(
        "--number_of_subproblems", type=int, help="Number of subproblems."
    )
    parser.add_argument("--steps", type=int, help="Number of splitting.")
    parser.add_argument("--verbose", action="store_true", help="Verbose.")

    parser.add_argument(
        "--problem_name", type=str, help="Which problem to solve.",
    )
    parser.add_argument(
        "--method_prompt",
        type=str,
        choices=["standard", "cot"],
        help="Which prompting strategy to use.",
    )
    parser.add_argument(
        "--l2m", action="store_true", help="Whether to use Least-to-Most prompting."
    )
    return parser.parse_args()


def main(args):
    print(f"Model name : {args.model_name_or_path}")
    k = args.k
    arguments = {
        "model_name_or_path": args.model_name_or_path,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
    }

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "num_return_sequences": args.num_return_sequences,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample,
        "request_batch_size": args.request_batch_size,
        "verbose": args.verbose,
    }

    # Set the stop words for the generation
    stop_words = []
    stop_words += [
        "###",
        "\n" * 5,
        "\n\n---",
        "____",
        "....",
        ". . . .",
        "Q:",
        "\nProblem:",
        "://",
        "\nA:",
        "<|eot_id|>",
        "<|start_header_id|>",
        # "\n\nWe can conclude that",
        "\n\nProblem:",
        "\n\nInput:",
        "#include",
        "[INST]",
        "\nHuman:",
    ]
    # Get divide, solve and merge
    divide_fn = divide(problem_name=args.problem_name, l2m=args.l2m)
    get_solve_prompt = solve_function(
        problem_name=args.problem_name, description=args.method_prompt, k=k
    )
    get_merge_prompt = merge_function(problem_name=args.problem_name, l2m=args.l2m)

    if args.inference_api == "vllm":
        generator = vLLMGenerator(**arguments)
    elif args.inference_api == "openai":
        generator = OpenAIGenerator(api_key=args.api_key, **arguments)
    elif args.inference_api == "hf":
        generator = HFGenerator(**arguments)
    else:
        pass

    def generate(prompts: List[str], max_tokens: int = args.max_new_tokens, n: int = 1):
        generation_kwargs["max_new_tokens"] = max_tokens
        generation_kwargs["num_return_sequences"] = n
        outputs = generator.generate(prompts=prompts, **generation_kwargs)
        outputs = [
            [_stop_at_stop_token(element, stop_words).strip() for element in output]
            for output in outputs
        ]
        return outputs

    # Get the solve function
    def solve(
        sentences: List[str],
        max_tokens: int = args.max_new_tokens,
        n: int = args.num_return_sequences,
    ):
        prompts = [
            apply_template(args.model_name_or_path)(get_solve_prompt(sentence))
            for sentence in sentences
        ]
        if len(prompts) > 0:
            print(f"===Solve\n{prompts[0]}\n===")
        outputs = generate(prompts=prompts, max_tokens=max_tokens, n=n)
        outputs_list = []
        for output in outputs:
            outputs_list.append(
                get_best_sentence(
                    sentences=output,
                    problem_name=args.problem_name,
                    verbose=args.verbose,
                )
            )
        return outputs_list

    # Get the merge function
    def merge(sentences: List[str], inputs: List[List[str]], outputs: List[List[str]]):
        """
        Build a few-shot for the merge of the subproblems.
        Arguments
        ---------
            -  sentences: List[str],
                Sequence that was be decomposed
            - inputs : List[List[str]],
                List of subproblems derived from each problem.
            - outputs : List[List[str]],
                List of the subproblems' solutions.
        """
        prompts = get_merge_prompt(sentences, inputs, outputs)
        prompts = [
            apply_template(args.model_name_or_path)(prompt) for prompt in prompts
        ]
        if len(prompts) > 0:
            print(f"===Merge\n{prompts[0]}\n===")
        outputs = generate(prompts)
        outputs_list = [output[0] for output in outputs]
        return outputs_list

    # Get the questions to solve
    from top.dataset import get_dataset

    pairs = get_dataset(args.problem_name, args.dataset_name_or_path)
    questions = [a for (a, _) in pairs]
    questions = (
        questions[: args.max_samples] if args.max_samples is not None else questions
    )

    if args.metadata_dir:
        metadata_dir = args.metadata_dir
    else:
        depth = args.steps
        breadth = args.number_of_subproblems
        if breadth >= 0:
            metadata_dir = f"{args.problem_name}_{args.method_prompt}_{k}_shot_seed_{args.seed}_{breadth}_{depth}"
        else:
            metadata_dir = f"{args.problem_name}_{args.method_prompt}_{k}_shot_seed_{args.seed}_None_{depth}"
        if args.num_return_sequences >= 2:
            metadata_dir += (
                f"_SC_{args.num_return_sequences}"  # temperature = 0.7, top_p = 0.95
            )

    output_dir = args.output_dir
    output_dir = os.path.join(output_dir, args.model_name_or_path.split("/")[-1])
    output_dir = os.path.join(output_dir, args.problem_name)
    os.makedirs(output_dir, exist_ok=True)

    metadata_dir = os.path.join(output_dir, metadata_dir)
    os.makedirs(metadata_dir, exist_ok=True)

    print(f"There are {len(questions)} samples!")
    # main problems, problems for round 1, ..., problems for round N
    list_of_sentences = [questions]
    # parents, parents for round 1, ..., parents for round N
    list_of_dictionaries = [{}]
    # Go trough the number of dividing rounds
    if args.number_of_subproblems >= 2 or args.number_of_subproblems == 0:
        for round in tqdm(range(args.steps)):
            list_of_propositions = []
            if os.path.exists(os.path.join(metadata_dir, f"divide_{round+1}.jsonl")):
                print(
                    f"Reading from {os.path.join(metadata_dir, f'divide_{round+1}.jsonl')}!"
                )
                with open(
                    os.path.join(metadata_dir, f"divide_{round+1}.jsonl"), "r"
                ) as fin:
                    for line in fin:
                        list_of_propositions.append(json.loads(line)["propositions"])
                dico = {}
                with open(
                    os.path.join(metadata_dir, f"parent_{round+1}.jsonl"), "r"
                ) as fin:
                    for line in fin:
                        dico = json.loads(line)
                # Number of keys in dico indicates the number of sentences that have already been divided
                start = 1 + max([v for _, v in dico.items()]) if len(dico) != 0 else 0
                print(f"Resuming from index {start}.")
            else:
                start = 0
            # Resume the division were it stopped
            sentences = list_of_sentences[-1]
            resume_list_of_propositions = divide_fn(
                sentences[start:], n_splits=args.number_of_subproblems
            )
            list_of_propositions.extend(resume_list_of_propositions)
            with open(
                os.path.join(metadata_dir, f"divide_{round+1}.jsonl"), "a"
            ) as fout:
                for j in range(start, len(sentences)):
                    fout.write(
                        json.dumps(
                            {
                                "sentence": sentences[j],
                                "propositions": list_of_propositions[j],
                            }
                        )
                        + "\n"
                    )
            dico = {}
            key = 0
            for a, propositions in enumerate(list_of_propositions):
                for b in range(len(propositions)):
                    dico[key + b] = a
                key += len(propositions)
            with open(
                os.path.join(metadata_dir, f"parent_{round+1}.jsonl"), "w"
            ) as fout:
                fout.write(json.dumps(dico))

            # Sentences for the next round are the propositions of the current round
            sentences = [
                prop for propositions in list_of_propositions for prop in propositions
            ]
            list_of_sentences.append(sentences)
            list_of_dictionaries.append(dico)

        max_steps = args.steps
    else:
        sentences = questions
        dict_of_sentences = {round: [] for round in range(args.steps)}
        dict_of_parents = {round: {} for round in range(args.steps)}
        if all(
            [
                os.path.exists(os.path.join(metadata_dir, f"divide_{round+1}.jsonl"))
                for round in range(args.steps)
            ]
        ):
            for round in range(args.steps):
                list_of_propositions = []
                with open(
                    os.path.join(metadata_dir, f"divide_{round+1}.jsonl"), "r"
                ) as fin:
                    for line in fin:
                        list_of_propositions.append(json.loads(line)["propositions"])
                dico = {}
                with open(
                    os.path.join(metadata_dir, f"parent_{round+1}.jsonl"), "r"
                ) as fin:
                    for line in fin:
                        dico = json.loads(line)
                # Number of keys in dico indicates the number of sentences that have already been divided
                start = 1 + max([v for _, v in dico.items()]) if len(dico) != 0 else 0
                dico = {int(k): int(v) for k, v in dico.items()}
                print(f"Resuming from index {start}.")
                dict_of_sentences[round] += sum(list_of_propositions, [])
                dict_of_parents[round] = dico
        else:
            start = 0
        resume_list_of_propositions = divide_fn(sentences[start:], n_splits=args.steps)
        for i, element in enumerate(resume_list_of_propositions):
            assert (
                len(element) == args.steps
            ), f"Each subdivision should be of length {args.steps}. Got {len(element)} instead. Check `divide_fn`"
            for j in range(len(element)):
                # First in element = solved first
                dict_of_sentences[j].append(element[args.steps - 1 - j])
                dict_of_parents[j][i + start] = i + start

        for round in range(args.steps):
            with open(
                os.path.join(metadata_dir, f"divide_{round+1}.jsonl"), "a"
            ) as fout:
                for j in range(start, len(sentences)):
                    fout.write(
                        json.dumps(
                            {
                                "sentence": sentences[j],
                                "propositions": [dict_of_sentences[round][j]],
                            }
                        )
                        + "\n"
                    )
            with open(
                os.path.join(metadata_dir, f"parent_{round+1}.jsonl"), "w"
            ) as fout:
                fout.write(json.dumps(dict_of_parents[round]))

        list_of_sentences = [dict_of_sentences[round] for round in range(args.steps)]
        list_of_dictionaries = [dict_of_parents[round] for round in range(args.steps)]
        max_steps = args.steps - 1

    step = max_steps
    previous_solutions = None
    while step >= 0:
        sentences = list_of_sentences[step]
        # Resume where we stopped
        current_solutions = []
        if os.path.exists(os.path.join(metadata_dir, f"answer_{step}.jsonl")):
            with open(os.path.join(metadata_dir, f"answer_{step}.jsonl"), "r") as fin:
                for line in fin:
                    current_solutions.append(json.loads(line)["output"])

        start = len(current_solutions)
        if step == max_steps:
            # We are at the leaves of the tree, we should use the solver
            for i in tqdm(range(start, len(sentences), args.request_batch_size)):
                inputs = sentences[i : min(i + args.request_batch_size, len(sentences))]
                outputs = solve(inputs)
                # Save the predictions to an output file
                with open(
                    os.path.join(metadata_dir, f"answer_{step}.jsonl"), "a"
                ) as fout:
                    for j, output in enumerate(outputs):
                        current_solutions.append(output)
                        fout.write(
                            json.dumps(
                                {
                                    "sentence": sentences[i + j],
                                    "output": output.strip(),
                                }
                            )
                            + "\n"
                        )
        else:
            # We are not at the leaves of the tree, we should solve each problem instance based on the next level's instances + solutions
            assert (
                previous_solutions is not None
            ), f"previous_solutions ({previous_solutions}) is None"
            inputs = [[] for _ in range(len(sentences))]
            outputs = [[] for _ in range(len(sentences))]
            if args.l2m:
                # Least-to-Most Prompting
                # Put the solution of all the previous problems in the right order.
                for w in range(max_steps + 1, step, -1):
                    if os.path.exists(os.path.join(metadata_dir, f"answer_{w}.jsonl")):
                        with open(
                            os.path.join(metadata_dir, f"answer_{w}.jsonl"), "r"
                        ) as fin:
                            for m, line in enumerate(fin):
                                inputs[m].append(json.loads(line)["sentence"])
                                outputs[m].append(json.loads(line)["output"])
                    else:
                        print(
                            f"{os.path.join(metadata_dir, f'answer_{w}.jsonl')} does not exist!"
                        )
            else:
                dico = list_of_dictionaries[step + 1]
                for key in dico:
                    inputs[dico[key]].append(list_of_sentences[step + 1][key])
                    outputs[dico[key]].append(previous_solutions[key])
            outputs = merge(
                sentences=sentences[start:],
                inputs=inputs[start:],
                outputs=outputs[start:],
            )
            # Save the predictions to an output file
            with open(os.path.join(metadata_dir, f"answer_{step}.jsonl"), "a") as fout:
                for j, output in enumerate(outputs):
                    current_solutions.append(output)
                    fout.write(
                        json.dumps(
                            {
                                "sentence": sentences[start + j],
                                "output": output.strip(),
                            }
                        )
                        + "\n"
                    )
        # Update parameters
        step -= 1
        previous_solutions = current_solutions

    print("END")


if __name__ == "__main__":
    args = parse_args()
    main(args)