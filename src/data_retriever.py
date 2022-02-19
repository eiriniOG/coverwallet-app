import pandas as pd

from .config import DICT_NULL_VALUES, DICT_MAJOR_VALUES, grouping_residuals, string_to_nomenclature

class DataRetriever():
    """
    This class process raw data and prepares it before predictions
    """

    def __init__(self, path_accounts, path_quotes, model: str):
        self.path_accounts = path_accounts
        self.path_quotes = path_quotes
        self.model = model

    def retrieve_appdata(self):
        self.raw_accounts, self.raw_quotes = self._get_data()

        #Clean raw data
        self._correct_types()
        self._no_duplicates()
        self._treat_nulls()

        #Return full dataset
        ret = self._create_master_data()
        return ret
    
    def prepare_variables(self,dataset):

        ds_grouped = self._grouping_variables(dataset)

        ds_dummies = ds_grouped
        if self.model != "catboost":
            ds_dummies = self._create_dummies(ds_grouped)
            
        return self._update_variables(ds_dummies)
        
    def _get_data(self):
        return pd.read_csv(self.path_accounts), pd.read_csv(self.path_quotes)

    def _correct_types(self):
        self.raw_quotes["carrier_id"] = self.raw_quotes["carrier_id"].astype(str)
    
    def _no_duplicates(self):
        self.raw_quotes = self.raw_quotes[~self.raw_quotes.duplicated()]

    def _treat_nulls(self):
        self.raw_accounts.fillna(value=DICT_NULL_VALUES, inplace=True)

    def _create_master_data(self):
        self.raw_quotes = self.raw_quotes.rename({"account_uuid": "company_id"}, axis="columns")
        aux = self.raw_accounts.merge(self.raw_quotes, how="inner", left_on=["account_uuid"],right_on=["company_id"])
        aux.drop(["company_id", "subindustry"], axis=1, inplace=True)
        return aux
    
    def _grouping_variables(self, dataset):
        dataset['product_res'] = dataset['product'].apply(lambda x: grouping_residuals(x, DICT_MAJOR_VALUES["major_products"]))
        dataset['carrier_id_res'] = dataset['carrier_id'].apply(lambda x: grouping_residuals(x, DICT_MAJOR_VALUES["major_carriers"]))
        dataset['state_res'] = dataset['state'].apply(lambda x: grouping_residuals(x, DICT_MAJOR_VALUES["major_states"]))
        dataset['industry_res'] = dataset['industry'].apply(lambda x: grouping_residuals(x, DICT_MAJOR_VALUES["major_industries"]))
        dataset['business_structure_res'] = dataset['business_structure'].apply(lambda x: grouping_residuals(x, DICT_MAJOR_VALUES["major_bu_structures"]))
        dataset.drop(["product", "carrier_id", "state", "industry", "business_structure"], axis=1, inplace=True)
        return dataset

    def _create_dummies(self, dataset):
        name_vars_object = [name for name,tipo in dataset.dtypes.iteritems() if 'object' in str(tipo) and name not in ["account_uuid"]]
        dataset[name_vars_object] = dataset[name_vars_object].applymap(lambda x: string_to_nomenclature(x))

        #Building dummies in a separate dataset
        return pd.get_dummies(dataset,columns=name_vars_object)
    
    def _update_variables(self, dataset):
        dataset["company_age"] = pd.datetime.now().year - dataset["year_established"]
        if self.model != "catboost":
            dataset.drop(["year_established", "total_payroll", "carrier_id_res_53", "business_structure_res_individual"], axis=1, inplace=True)
        else:
            dataset.drop(["year_established", "total_payroll"], axis=1, inplace=True)
        return dataset