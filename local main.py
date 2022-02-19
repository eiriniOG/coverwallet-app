from src.accvalue_app import AccValueApp
import argparse

def main():

    parser = argparse.ArgumentParser(description='Make account value predictions with a pretrained model')
    parser.add_argument('model', type=str, default="randomforest_cal", choices=['xgboost_cal', 'logregression', 'catboost', 'randomforest_cal'])
    args = vars(parser.parse_args())

    # self.path_quotes = os.path.join(r'C:\Users\Irene\Desktop\coverwallet-test\statement', "quotes_test.csv")
    # self.path_accounts = os.path.join(r'C:\Users\Irene\Desktop\coverwallet-test\statement', "accounts_test.csv")

    app = AccValueApp(args["model"])
    res = app.run()
    print(res.to_json())



main()