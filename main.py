import argparse
from src.accvalue_app import AccValueApp


def main():

    parser = argparse.ArgumentParser(description='Make account value predictions with a pretrained model')
    parser.add_argument('model', type=str, default="randomforest_cal", choices=['xgboost_cal', 'logregression', 'catboost', 'randomforest_cal'])
    args = vars(parser.parse_args())

    app = AccValueApp(args["model"])
    res = app.run()
    print(res.head())

main()