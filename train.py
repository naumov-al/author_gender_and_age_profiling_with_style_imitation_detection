import argparse as ap
import json
import os
from NgramModels import CharNgramClassifier
from NNModel_simple import NNModelSimple
from corpus_tools import get_y, balance_data

if __name__ == "__main__":
    args_parser = ap.ArgumentParser(description="The train classifier module.")
    args_parser.add_argument("train_data", help="Path to train data in JSON format")
    args_parser.add_argument("out_model_file", help="Path for the result model")
    args_parser.add_argument("--y-field", dest="y", default="gender")
    args_parser.add_argument("--balance-data", dest="balance_data", default=False, action="store_const", const=True)
    args_parser.add_argument("--balance-data-stratege", dest="balance_data_strategey", help="{undersampling, oversampling}")
    args_parser.add_argument("--model-type", default="CharNgram", help={"CharNgram, NNModel"})
    args = args_parser.parse_args()

    with open(os.path.expandvars(args.train_data), "r") as f:
        inp = json.load(f)
    y = get_y(inp, args.y)

    if args.balance_data:
        inp, y = balance_data(inp, y, strategy="undersampling")

    if args.model_type == "CharNgram":
        model = CharNgramClassifier()
        model.fit(inp, y)
        model.save_model(os.path.expandvars(args.out_model_file))
    elif args.model_type == "NNModel":
        model = NNModelSimple(args.out_model_file)
        model.fit(inp, y)
        model.save_model(args.out_model_file)

    print("Done!")