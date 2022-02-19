import pickle
import pandas as pd

class ConversionModel():
    """
    This class predicts final account value for each account id
    """

    def __init__(self, path_model):
        self.model = pickle.load(open(path_model, 'rb'))

    def predict(self, dataset):
        results = dataset[["account_uuid", "premium_orig"]]
        results["prob"] = [pair[1] for pair in self.model.predict_proba(dataset.drop(["account_uuid", "premium_orig"], axis=1))]
        results["pred"] = self.model.predict(dataset.drop(["account_uuid", "premium_orig"], axis=1))
        #right side of the tuple has the probability of conversion = 1
        return results

    def calculate_accvalue(self, results):
        results["exp_accvalue"] = results["prob"] * results["premium_orig"]
        return pd.DataFrame(results[["account_uuid", "exp_accvalue"]].groupby("account_uuid").agg({"exp_accvalue": sum})).reset_index()
