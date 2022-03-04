import os
from .data_retriever import DataRetriever
from .scaler import DataScaler
from .predictor import ConversionModel
from .config import ASSETS_PATH, MODEL_NAMES

class AccValueApp():
    """
    This class manages the end-to-end process of conversion prediction and account value calculation
    """

    def __init__(self):
        
        self.__load_assets()

    def run(self, model_name, accountsFile, quotesFile):

        model = self.model_dict[model_name]

        #Get data
        retr_data = DataRetriever(accountsFile,quotesFile, model_name)
        ds_input = retr_data.retrieve_appdata()
        ds_input_treated = retr_data.prepare_variables(ds_input)

        #Scale data
        ds_scaled = self.scaler.scale(ds_input_treated, model_name)

        #Predict
        results_pred = model.predict(ds_scaled)
        return model.calculate_accvalue(results_pred)

    def __load_assets(self):

        self.model_dict = {}
        self.scaler = DataScaler(os.path.join(ASSETS_PATH, "robustscaler_pkl"))
        for model in MODEL_NAMES:
            self.model_dict[model] = ConversionModel(os.path.join(ASSETS_PATH, model + "_pkl"))

        