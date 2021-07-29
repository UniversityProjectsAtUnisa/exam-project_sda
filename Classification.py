from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np
from utils import Utils
from matplotlib import pyplot as plt

# ==============================   CONFIG     ============================

ABS_PATH = '/mnt/c/Users/marco/Documents/UNISA/SDA/progetto/SDAgruppo2'
DATASET_FILENAME = 'ClassificationData_SDA_AH_group2.csv'
Y_LABEL = 'Z_OppositeTeamDefence'
PREDICTORS_NUMBER = 8

# ================================ START =================================

utils = Utils(ABS_PATH, DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)
df = utils.read_dataset()

# ======================= BASE LOGISTIC REGRESSION =======================

# lr, X, y = utils.logisticRegression(df)
# scores = utils.inspectLr(lr, X, y, k=5)

utils.plot_heatmap(df)

# print(scores)


# # ======================= BASE LOGISTIC REGRESSION =======================


# df = pd.read_csv('ClassificationData_SDA_AH_group2.csv')
# print(df)

# X = df.loc[:, df.columns != 'Z_OppositeTeamDefence'].to_numpy()
# y = df['Z_OppositeTeamDefence'].to_numpy()

# X_train = df.loc[:149, df.columns != 'Z_OppositeTeamDefence'].to_numpy()
# y_train = df['Z_OppositeTeamDefence'][:150].to_numpy()

# X_test = df.loc[150:, df.columns != 'Z_OppositeTeamDefence'].to_numpy()
# y_test = df['Z_OppositeTeamDefence'][150:].to_numpy()


# lr = LogisticRegression().fit(X, y)
# scores = cross_validate(lr, X, y, n_jobs=5, cv=5)
# # Bisogna mettere accuracy?
# avg_score = sum(scores["test_score"])/len(scores["test_score"])
