import argparse as ap
import json
import os
from NgramModels import CharNgramClassifier
from NNModel_simple import NNModelSimple
from corpus_tools import get_y, balance_data


def add_age_group(inp, l_age_groups=[(20, 30), (30, 40), (40, 50)]):
    for doc in inp:
        doc["meta"]["age_group"] = "Other"
        doc_age = doc["meta"].get("age", None)
        if doc_age is not None:
            for age_group_min, age_group_max in l_age_groups:
                if age_group_min <= doc_age < age_group_max:
                    doc["meta"]["age_group"] = "{0}-{1}".format(age_group_min, age_group_max)
                    break
    return inp


if __name__ == "__main__":
    args_parser = ap.ArgumentParser(description="The train classifier module.")
    args_parser.add_argument("train_data", help="Path to train data in JSON format")
    args_parser.add_argument("out_model_file", help="Path for the result model")
    args_parser.add_argument("--y-field", dest="y", default="gender")
    args_parser.add_argument("--balance-data", dest="balance_data", default=False, action="store_const", const=True)
    args_parser.add_argument("--balance-data-stratege", dest="balance_data_strategey", help="{undersampling, oversampling}")
    args_parser.add_argument("--model-type", default="CharNgram", help={"CharNgram, NNModel"})
    args_parser.add_argument("--add-age-group", default=False, const=True, action="store_const")
    args_parser.add_argument("--age-group-no-other", default=False, const=True, action="store_const")
    args = args_parser.parse_args()

    with open(os.path.expandvars(args.train_data), "r") as f:
        inp = json.load(f)

    if args.add_age_group:
        inp = add_age_group(inp)
    if args.age_group_no_other:
        inp = [doc for doc in inp if doc["meta"]["age_group"] != "Other"]

    y = get_y(inp, args.y)

    if args.balance_data:
        inp, y = balance_data(inp, y, strategy="undersampling")

    if not os.path.exists(args.out_model_file):
        if not args.out_model_file.endswith(".pkl"):
            os.makedirs(args.out_model_file)

    if args.model_type == "CharNgram":
        model = CharNgramClassifier()
        model.fit(inp, y)
        model.save_model(os.path.expandvars(args.out_model_file))
    elif args.model_type == "NNModel":
        model = NNModelSimple(args.out_model_file)
        model.fit(inp, y)
        model.save_model(args.out_model_file)

    print("Done!")