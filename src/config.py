import os

DICT_NULL_VALUES = {"industry": "blank", "subindustry": "blank", "state": "FL", "year_established": 2017, "annual_revenue": 217706.9, "total_payroll": 43469.87, "num_employees": 8}
DICT_MAJOR_VALUES = {
"major_bu_structures" : ["Limited Liability Company", "Individual", "Corporation"],
"major_products" : ["General_Liability", "Business_Owners_Policy_BOP", "Workers_Compensation", "CW_Professional_Liability", "Commercial_Auto", "CW_Errors_Omissions_E_O", "CW_Umbrella_Policy", "Package"],
"major_carriers" : ["39", "31", "60","30","29","9","53","21","73","22","40","72"],
"major_states" : ["FL","CA","NY","TX","GA","NJ","VA","PA","NC","SC","IL","LA","MA","OH","MI","CO","MD","AL","TN","MO","AZ","WA","WI","IN","NV","OK","KY","MS","OR"],
"major_industries" : ["Contractors", "Retail Trade", "Other Services", "Professional, Scientific and Technical Services", "Administrative Services and Building Maintenance", "Food and Accommodation", "Consultants", "Technology, Media and Telecommunications", "Manufacturing", "Healthcare", "Wholesale Trade", "Transportation and Warehousing", "Real Estate", "Sports, Arts, Entertainment, and Recreation", "Construction", "Finance and Insurance", "Education" ]
}

model_features = ['premium', 'annual_revenue', 'company_age', 'num_employees', 'carrier_id_res_60', 'business_structure_res_limited_liability_company', 'product_res_business_owners_policy_bop', 'carrier_id_res_39', 'carrier_id_res_31', 'industry_res_other_services', 'business_structure_res_corporation', 'state_res_fl', 'industry_res_retail_trade', 'product_res_general_liability', 'state_res_ca', 'industry_res_contractors', 'state_res_residuals', 'industry_res_food_and_accommodation', 'industry_res_professional__scientific_and_technical_services', 'state_res_ny', 'state_res_tx', 'state_res_ga', 'industry_res_administrative_services_and_building_maintenance', 'industry_res_consultants', 'business_structure_res_residuals', 'state_res_va', 'industry_res_manufacturing', 'state_res_nj', 'state_res_nc', 'state_res_il', 'state_res_sc', 'carrier_id_res_residuals', 'carrier_id_res_30', 'carrier_id_res_21', 'carrier_id_res_9', 'state_res_la', 'state_res_pa', 'state_res_mi', 'industry_res_technology__media_and_telecommunications', 'carrier_id_res_29', 'state_res_oh', 'state_res_tn', 'industry_res_wholesale_trade', 'industry_res_healthcare', 'product_res_residuals', 'state_res_md', 'state_res_co', 'state_res_al', 'state_res_ma', 'state_res_mo', 'state_res_az', 'state_res_wi', 'product_res_workers_compensation', 'industry_res_education', 'industry_res_residuals', 'product_res_cw_professional_liability', 'product_res_commercial_auto', 'state_res_nv']
model_features_cat = ["premium","annual_revenue","num_employees","product_res","carrier_id_res","state_res" ,"industry_res" ,"business_structure_res","company_age"]
features_to_standardize = ["premium","annual_revenue","company_age","num_employees"]

ASSETS_PATH = os.path.dirname(os.path.abspath(__file__)) + r'/../assets'
MODEL_NAMES = ["catboost", "logregression", "randomforest_cal", "gboost_cal"] #"xgboost_cal"

def grouping_residuals(var_value: str, major_values: list) -> str:
  if var_value in major_values:
    return var_value
  else:
    return "residuals"

def string_to_nomenclature(text):
    return text.strip().replace(',','_').replace('/','_').replace('.','_').replace(' ','_').replace('-','_').replace('(','_').replace(')','_').lower()