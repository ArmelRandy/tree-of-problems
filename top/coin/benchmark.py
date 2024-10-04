import numpy as np
import pandas as pd
import argparse
from faker import Faker

from top.coin import DATA_PATH
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--number_of_samples", type=int, help="Number of samples in the benchmark"
    )
    parser.add_argument(
        "--heads_up_probability",
        type=float,
        default=1.0,
        help="Probability that coin starts as heads up.",
    )
    parser.add_argument("--seed", type=int, help="seed")
    parser.add_argument(
        "--k",
        type=int,
        help="Number of people that either do flip or don't flip the coin.",
    )
    return parser.parse_args()


def main(args):
    fake = Faker()
    Faker.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    samples = []
    # 1, 2, 3, 4, 5, 6, 7, 8, 9
    for i in range(args.number_of_samples):
        question = ""
        if rng.uniform() <= args.heads_up_probability:
            start = "heads"
        else:
            start = "tails"
        status = True
        question += f"A coin is {start} up. "
        if args.k is None:
            k = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        else:
            k = args.k
        for _ in range(k):
            name = fake.first_name()
            if rng.uniform() < 0.5:
                question += f"{name} flips the coin. "
                status = not status
            else:
                question += f"{name} does not flip the coin. "
        question += f"Is the coin still {start} up?"
        answer = "yes" if status else "no"
        samples.append((question, answer))
    df = pd.DataFrame(
        {"Question": [a for (a, _) in samples], "Answer": [b for (_, b) in samples]}
    )
    df.to_csv(os.path.join(DATA_PATH, f"coin_{args.k}.csv"), index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
