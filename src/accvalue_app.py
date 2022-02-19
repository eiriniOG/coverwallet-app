import os
from .data_retriever import DataRetriever
from .scaler import DataScaler
from .predictor import ConversionModel

class AccValueApp():
    """
    """

    def __init__(self, model):

        self.path_model = os.path.join(r'C:\Users\Irene\Desktop\coverwallet-test\assets', model + "_pkl")
        self.path_scaler = os.path.join(r'C:\Users\Irene\Desktop\coverwallet-test\assets', "robustscaler_pkl")
        self.path_quotes = os.path.join(r'C:\Users\Irene\Desktop\coverwallet-test\statement', "quotes_test.csv")
        self.path_accounts = os.path.join(r'C:\Users\Irene\Desktop\coverwallet-test\statement', "accounts_test.csv")

    def run(self):

        #Get data
        retr_data = DataRetriever(self.path_accounts,self.path_quotes, self.path_model)
        ds_input = retr_data.retrieve_appdata()
        ds_input_treated = retr_data.prepare_variables(ds_input)

        #Scale data
        scaler = DataScaler(self.path_scaler, self.path_model)
        ds_scaled = scaler.scale(ds_input_treated)

        #Predict
        model = ConversionModel(self.path_model)
        results_pred = model.predict(ds_scaled)
        return model.calculate_accvalue(results_pred)