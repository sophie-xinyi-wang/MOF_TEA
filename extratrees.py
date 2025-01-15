"""
EDT model example

This script demonstrates how to create, train, and save a extremely randomized decision trees model using scikit-learn.
It uses the data from a csv file for training, validating and testing the model. 

""" 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import make_scorer, mean_squared_error
import sklearn
import os
from sklearn.inspection import permutation_importance
from mlutility import predict_with_model
import pickle

model_name = "extratrees"
outdir = './' + model_name
if not os.path.exists(outdir):
    os.mkdir(outdir)

# Get data from csv file
dataset = pd.read_csv("combined_data.csv")
dataset.pop('Name')
X = dataset[['Temperature(K)', 'Pressure(bar)', 'Density', 'GSA', 'LCD', 'PLD', 'PV', 'VF', 'VSA']] #inputs
y = dataset['Excess Uptake(mg/g)']
feature_names = dataset.columns.tolist()[:-4]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)

estimator = ExtraTreesRegressor()
# Define the grid of hyperparameters to search
param_grid = {
    'criterion' : ["squared_error"],
    'n_estimators': [5, 10, 20],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 8],
    'min_samples_leaf': [2, 4, 6],
    'max_features': ['sqrt', 'log2',None,4]
}
scoring = make_scorer(r2_score, greater_is_better=True)

# Use grid search to find best parameters
grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=scoring, cv=8, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
df_cv_results = pd.DataFrame(grid_search.cv_results_)
df_cv_results.to_csv(os.path.join(outdir, model_name + 'CvResult.csv'))
print("Best parameters found: ", best_params)


# Now that we have the best params, use them to re-train the training dataset, then evaluate model by test dataset.
best_model = ExtraTreesRegressor(**best_params)
best_model.fit(X_train, y_train)
# Calculate feature importance using permutation importance
importances = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
perm_importances = importances.importances_mean
features = feature_names
perm_importance_df = pd.DataFrame({'Feature': features, 'Permutation Importance': perm_importances}).sort_values(by='Permutation Importance', ascending=False)
perm_importance_df.to_csv(os.path.join(outdir, model_name + '_importances.csv'))
predict_with_model(model=best_model, X_test=X_test, X_train=X_train, y_test=y_test, y_train=y_train,filename=os.path.join(outdir, model_name))
with open(os.path.join(outdir, model_name)+'_model.pkl','wb') as file:
    pickle.dump(best_model, file)
