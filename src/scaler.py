import pickle

from .config import model_features, model_features_cat, features_to_standardize

class DataScaler():

    """
    This class scales selected features
    """

    def __init__(self, path_scaler):
        self.scaler = pickle.load(open(path_scaler, 'rb'))
    
    def _select_features(self, dataset, model_name):
        if model_name != "catboost":
            ret = dataset[["account_uuid"] + model_features]
        else:
            ret = dataset[["account_uuid"] + model_features_cat]
        
        ret["premium_orig"] = ret["premium"]

        return ret
        
    def scale(self, dataset, model_name:str):
        aux = self._select_features(dataset, model_name)
        aux[features_to_standardize] = self.scaler.transform(aux[features_to_standardize])
        return aux
    