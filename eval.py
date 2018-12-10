import numpy as np
import pandas as pd
import argparse as ap
import os
import json
from sklearn.metrics import classification_report, accuracy_score
from corpus_tools import balance_data, get_y
from NgramModels import CharNgramClassifier
from NNModel_simple import NNModelSimple


if __name__ == "__main__":
    args_parser = ap.ArgumentParser()
    args_parser.add_argument("gold_y_field")
    args_parser.add_argument("gold_json")
    args_parser.add_argument("model")
    args_parser.add_argument("--model-type", default="CharNgram")
    args_parser.add_argument("--balance-data", default=False, const=True, action="store_const")
    args = args_parser.parse_args()

    with open(os.path.expandvars(args.gold_json), "r") as f:
        inp = json.load(f)

    y_gold = [val for val in get_y(inp, args.gold_y_field)]
    if args.balance_data:
        inp, y_gold = balance_data(inp, y_gold)

    if args.model_type == "CharNgram":
        model = CharNgramClassifier.load_model(args.model)
        pred = model.predict(inp)
    elif args.model_type == "NNModel":
        model = NNModelSimple(args.model)
        model.load_model(args.model)
        pred = model.predict(inp)
        y_gold = [str(val) for val in get_y(inp, args.gold_y_field)]

    print(classification_report(y_gold, pred))