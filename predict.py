import argparse as ap
import json
import os
from NgramModels import CharNgramClassifier
from NNModel_simple import NNModelSimple

if __name__ == "__main__":
    args_parser = ap.ArgumentParser(description="The train classifier module.")
    args_parser.add_argument("test_data", help="Path to train data in JSON format")
    args_parser.add_argument("model", help="Path for the result model")
    args_parser.add_argument("--model-type", default="CharNgram", help="{CharNgram, NNModel}")
    args = args_parser.parse_args()

    with open(os.path.expandvars(args.test_data), "r") as f:
        inp = json.load(f)

    if args.model_type == "CharNgram":
        model = CharNgramClassifier.load_model(args.model)
        pred = model.predict(inp)
    elif args.model_type == "NNModel":
        model = NNModelSimple(args.model)
        model.load_model(args.model)
        pred = model.predict(inp)

    for doc, pred_val in zip(inp, pred):
        print("Text:")
        print(doc["text"].strip())
        print("class: {0}".format(pred_val))
        print("")

    print("Done!")
