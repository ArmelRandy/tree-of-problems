import numpy as np
import pandas as pd
import argparse
import os

from top.algebraic import DATA_PATH


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--number_of_samples", type=int, help="Number of samples in the benchmark"
    )
    parser.add_argument(
        "--addition_probability",
        type=float,
        default=0.5,
        help="Probability that coin starts as heads up.",
    )
    parser.add_argument(
        "--negative_first_probability",
        type=float,
        default=0.5,
        help="Probability that the sample starts with a negative number",
    )
    parser.add_argument("--seed", type=int, default=122, help="seed")
    parser.add_argument(
        "--number_of_numbers",
        type=int,
        help="Number of numbers involved in the algebraic sum.",
    )
    parser.add_argument(
        "--number_of_digits",
        type=int,
        default=2,
        help="Number of digits in the numbers involved in the sum.",
    )
    return parser.parse_args()


def main(args):
    rng = np.random.default_rng(args.seed)
    samples = []
    for i in range(args.number_of_samples):
        if rng.uniform() <= args.negative_first_probability:
            sample = "-"
        else:
            sample = ""
        for j in range(args.number_of_numbers):
            s = ""
            s = str(rng.choice([k for k in range(1, 10)], size=1)[0])
            for _ in range(1, args.number_of_digits):
                digit = str(rng.choice([k for k in range(10)], size=1)[0])
                s += str(digit)
            if j != args.number_of_numbers - 1:
                if rng.uniform() <= args.addition_probability:
                    sample += f" {s} + "
                else:
                    sample += f" {s} - "
            else:
                sample += f"{s}"
        sample = sample.strip()
        sample = sample.replace("  ", " ")
        samples.append((sample, eval(sample)))
        df = pd.DataFrame(
            {"Question": [a for (a, _) in samples], "Answer": [b for (_, b) in samples]}
        )
    df.to_csv(
        os.path.join(
            DATA_PATH,
            f"algebraic_{args.number_of_numbers}_{args.number_of_digits}.csv",
        ),
        index=False,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
