# Coverwallet Codetest

## Problem description
I focused on finding a solution for the two challenges presented in the
 document `statement.pdf` by CoverWallet.
- **Challenge 1**: given data abour accounts and quotes, train a model to predict conversion and therefore account value (as defined in the statement).
- **Challenge 2**: serve the trained model through a consumable API.
 
 ## Tools
 - Python (see requirements.txt for dependencies)
 - Docker
 - Colab from Google
 
 ## Usage
 - **Challenge 1**: there are two key notebooks for this step in `/notebooks` folder:
 
 `model_training_v2.ipynb` where you will find all the steps involved in the model training and full details on decissions and conclusions. 
 
 `apply_model_test.ipynb` where you will find a manual script to apply trained models over new data. It is a functional-not-formal code to make quick submissions.
 
 Please take into account that this two are environment dependant, meaning that paths and libraries have to be modified and adjusted.
 - **Challenge 2**: it is resolved in the present repository through FastAPI. The fundamental code for formatting data and make predictions is in `/src` folder.
 
`docker build -t coverwalletapp .` will build the image

`docker run -p 80:80 coverwalletapp` will run the container. The app is launched via uvicorn. Refer to http://127.0.0.1:80/ for welcome and redirect to http://127.0.0.1:80/docs to try the prediction endpoint. You need to upload an accounts file (you may use "accounts_test.csv" located in `/notebooks/data`), a quotes file (you may use "quotes_test.csv" located in `/notebooks/data`), and specify a model from this list: "catboost", "logregression", "randomforest_cal", "gboost_cal", or ~"xgboost_cal"~ (currently experiencing some issues due to library version updates). Other values will result in response error. 

This should work on your side without further adjustments to the code.
  

 
 ## Solution
 ### Challenge 1: The Account Value Case Dilemma
 #### 1. Thinking
 When I first read the problem I thought about building a regression model, directly based on final account values. But that approach presented two problems: it ignores the existence of conversion middle step and it forces me to either aggregate quotes data somehow (not always feasible) or just discard it (and use only accounts information).
     
 #### 2. Developing
The fact conversion/not conversion is modeled through Classification techniques, and then it will be used to calculate expected account values according to the provided definition grouping by account ids. That means that the unit of modeling is a quote containing certain descriptive variable about the quote itself and the account that made it.

First of all, input data is processed and analyzed in order to find any possible issue, and also to transform it into a single master dataset.

Secondly, univariate and multivariate analysis are performed in order to solve data behaviours that might complicate statistical basis of training.

Modeling is then carried out, for five different algorithms: [Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees), [GBoost](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting), [XGBoost](https://xgboost.readthedocs.io/en/stable/index.html) and [CatBoost](https://catboost.ai/en/docs/).

Finally, testing the models over validation data gives a hint about which candidate should be picked as final model for prediction.
     

 #### 3. Insights and Results
 The following is a summary, please refer to notebook `model_training_v2.ipynb` for full details and plots.
##### 3.1 From Data
- Accounts initial dataset has 5709 rows and Quotes initial dataset has 11724 rows.
- Filtering was implemented when encountering duplicates in Quotes dataset (all full row duplicates were erased).
- Filtering was implemented when encoutering null values in Accounts dataset representing <1% of the total rows. That was the case of columns: state, year of establishment, annual revenue, total payroll, business structure and number of employees. Null values that may appear in the future will be interpreted as the mean for continuous variables or the most frequent value for categorical variables.
- In Accounts dataset, industry and subindustry null values represented almost 3% of rows, so they were substituted by "blank" new value. However, subindutry variable was finally discarded due to its large diversity.
- Residual variables (<1%) related to products, carriers, states and industries are grouped into a single category. A new variable "company age" is calculated from year of establishment (discarded).
- Continuous variables present a huge amount of outliers, which were filtered with an iqr expansion factor of 4 from Q3.
- Conversion cases are specially low for premiums between 500$-505$. General liability is the product with highest number of quotes, but Commercial Auto, CW Errors Omissions and Package have the highest conversion rates. California and Florida are the states with the higher number of quotes, but Kentucky and Oregon have the highest conversion rates.
- Variables with correlation higher than 70% have been analyzed and treated. A curious case was that Carrier id=53 was highly correlated with Commercial Auto product, so it might be a specialized insurance company with a high market quota in the US or Florida (maybe Progressive or Geico according to quick Google search).
- The final dataset has 9032 rows and is fairly balanced in terms of conversion (60% of 1s-40% of 0s).

##### 3.1 From Modeling
- 58 variables among continuous and dummies are pre-selected for modeling with a 80-20% train/validation split, meanwhile only 9 original variables are necessary for CatBoost.
- Due to the huge amount of outliers, RobustScaler is used to scale continuous variables before modeling and avoid lack of convergence in algorithms weak against outliers like LogisticRegression.
- F1-score is the preferred metric to assess model training and selection, since it is desirable to
>Refine true positives, in order to optimize further marketing actions resources to increase conversion.

>Avoid as many false negatives as possible, since the opposite would mean losing chances of potential clients.
- Below, the predictive robustness of all casuistics is described, for the best candidates from CrossValidation and GridSearch tuning.

|Model|Train F1score|Validation AUC|Validation F1score|Validation RMSE($)|
|:---|:----:|:----:|:----:|:----:|
|LogReg|75,40%|68,55%|76,02%|438,39|
|RandomForest|76,43%|63,77%|77,68%|396,25|
|Gboost|76,66%|67,13%|77,33%|408,43|
|XGBoost|76,66%|67,30%|77,28%|405,17|
|CatBoost|76,31%|68,59%|76,54%|420,90|

Submitted results over provided test data are calculated with RandomForest since it is the model with lowest RMSE. Nevertheless, expected account values (using predicted probabilites) are provided also for the other algoritms.
- The models are calibrated to adjust predicted probabilities to conversion observed distribution.



 
 ### Challenge 2: Serving the model
 #### 1. Thinking
This part was optional but I wanted to undertake it since it gives the opportunity to design and code, outside the notebook-like approach.

The usecase of the model prediction functionality was not very clear so I had two choices: single row predictions (e.g., for streaming/real time use) or batch prediction. I went for the batch usecase since it was very similar to the requested submission test files.

 #### 2. Developing
Since I am not a fully skilled Python software developper, I have chosen [FastAPI](https://fastapi.tiangolo.com/) framework because of its friendly learning curve approach and its simplicity. It allowed me to build something functional in short time.

I decided to separate the different steps of model application into different classes, in case the code has to be updated in the future. The `DataRetriever` class takes charge of processing input files and generate a single dataset for prediction, the `DataScaler` applies pre-trained robust scaling to continuous columns, and the `ConversionModel` class makes predictions on conversion and calculates final account values (it would allow to change the response and return 1s/0s for variable convert, since they are two separate methods). `AccValueApp` class is in charge of orchestrating all those steps.

Pretrained models are stored in `/assets` folder and they will be used by the app from there.

Error responses are the standard APIrest codes.

 #### 3. Results
The app takes an Accounts csv and a Quotes csv with **the exact same headers** as provided in the statement, and performs all necessary treatments before predicting expected account values with a specified pretrained model. It returns a json response with accounts ids and expected account value for each one.

 ## Conclusions and improvements
- It was a lot of fun reading, understanding and implementing
 the solutions. I felt motivated when I started unraveling the problem :)
 - I did not find any reference to data stationality since there were no variables related to time or timestamps whatsoever. It would be interesting to understand quotes also in a time-context, because otherwise it is not possible to know the time-framed validity of predicted conversion (is the conversion observed in the next 24 hours? in the next week?).
 - The predictive robustness of the trained models are not very optimal, I would like to have more data and time to process it, along with a Business Owner, to fully understand the conversion funnel.
 - There is a lot of hardcoding within the repository due to specific column names above all, I tried to minimize it including a config file. Besides, the app code is barely commented.
 - I should define classes of model types, responses and usecases following formal FastAPI philosophy to make the app scalable. I also should include unit testing at least.
 - Models are stored in a folder repository, this could be a problem in terms of heaviness. In a formal environment, models could be better invoked from other kind of storage (like cloud compartments).
- The notebook approach is a risk in terms of environments and
 reproducibility of results. I should find a more portable solution also for the training in order to follow a philosophy closer to MLOps.
 
 ## References
Already linked within the document.
