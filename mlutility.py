"""
Model Validation
----------------------------
This script creates a function for model validation.

"""

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

def predict_with_model(model, X_test, X_train, y_test, y_train, filename):
    # Evaluate model
    y_test_result = model.predict(X_test)
    y_train_result = model.predict(X_train)
    test_data = np.column_stack((X_test,y_test,y_test_result))
    df_test = pd.DataFrame(test_data)
    df_test.to_csv(filename + 'test.csv')
    train_data = np.column_stack((X_train,y_train,y_train_result))
    df_train = pd.DataFrame(train_data)
    df_train.to_csv(filename + 'train.csv')
    # Draw evalutaion results
    score = model.score(X_test, y_test)
    print("Test score: ", score)
    mae_train = round(mean_absolute_error(y_train, y_train_result),4)
    r_squared_train = round(r2_score(y_train, y_train_result),4)
    mae_test = round(mean_absolute_error(y_test, y_test_result),4)
    r_squared_test = round(r2_score(y_test, y_test_result),4)
    plt.figure (figsize=(8,8))
    plt.scatter (y_test, y_test_result,s=4, color = "0.55")
    p1=max(0, 0)
    p2=min(-8, -8)
    plt.plot ([p1,p2], [p1,p2], 'k--')
    plt.xlabel ('True excess uptake (mg/g)', fontsize =18)
    plt.ylabel ('Predicted excess uptake (mg/g)', fontsize =18)
    plt.axis ('equal')
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.xlim([0,30])
    plt.ylim([0,30])
    plt.grid (visible=bool, axis='both',color ='k', linestyle = '-', linewidth =0.5 )
    plt.text(8, 28, 'MAE ='+ str(mae_test), horizontalalignment='center', verticalalignment='center', fontsize =20)
    plt.text(8, 23, 'R_sqaured ='+ str(r_squared_test), horizontalalignment='center', verticalalignment='center', fontsize =20)
    plt.savefig(filename + 'test_excess_uptake.png')
    plt.show()

    