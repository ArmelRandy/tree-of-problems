import numpy as np
import pandas as pd
import argparse
from faker import Faker

from top.concatenation import DATA_PATH
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--number_of_samples", type=int, help="Number of samples in the benchmark"
    )
    parser.add_argument("--seed", type=int, help="seed")
    parser.add_argument(
        "--k",
        type=int,
        help="Number of words whose last letter should be concatenated.",
    )
    return parser.parse_args()


def main(args):
    fake = Faker()
    Faker.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    samples = []
    # 1, 2, 3, 4, 5, 6, 7, 8, 9
    for i in range(args.number_of_samples):
        if args.k is None:
            k = rng.choice([4, 5, 6, 7, 8, 9, 10, 11, 12])
        else:
            k = args.k
        names = [fake.first_name() for _ in range(k)]
        question = ", ".join(names)
        answer = "".join([name[-1] for name in names])
        samples.append((question, answer))
    df = pd.DataFrame(
        {"Question": [a for (a, _) in samples], "Answer": [b for (_, b) in samples]}
    )
    df.to_csv(os.path.join(DATA_PATH, f"concatenation_{args.k}.csv"), index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
