# -*- coding: utf-8 -*-

import argparse
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input1")
    parser.add_argument("input2")
    parser.add_argument("output")

    args = parser.parse_args()

    pd.concat([pd.read_csv(f) for f in [args.input1, args.input2]])\
        .to_csv(args.output, index=False)
