import os
from .data_retriever import DataRetriever
from .scaler import DataScaler
from .predictor import ConversionModel

class AccValueApp():
    """
    This class manages the end-to-end process of conversion prediction and account value calculation
    """

    def __init__(self, model, accountsFile, quotesFile):
        assetsPath = os.path.dirname(os.path.abspath(__file__)) + r'/../assets'
        self.model = model
        self.path_model = os.path.join(assetsPath, self.model + "_pkl")
        self.path_scaler = os.path.join(assetsPath, "robustscaler_pkl")
        self.path_quotes = quotesFile
        self.path_accounts = accountsFile

    def run(self):

        #Get data
        retr_data = DataRetriever(self.path_accounts,self.path_quotes, self.model)
        ds_input = retr_data.retrieve_appdata()
        ds_input_treated = retr_data.prepare_variables(ds_input)

        #Scale data
        scaler = DataScaler(self.path_scaler, self.model)
        ds_scaled = scaler.scale(ds_input_treated)

        #Predict
        print(self.path_model)
        model = ConversionModel(self.path_model)
        results_pred = model.predict(ds_scaled)
        return model.calculate_accvalue(results_pred)