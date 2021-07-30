from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn import metrics
import pandas as pd
import numpy as np
import utils
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
# ==============================   CONFIG     ============================

ABS_PATH = r"C:\Users\carbo\OneDrive\Documenti\Magistrale Carbone\2 sem\Statistical Data Analysis\aPROGETTO\SDAgruppo2"
ABS_PATH = '/mnt/c/Users/marco/Documents/UNISA/SDA/progetto/SDAgruppo2'
DATASET_FILENAME = 'ClassificationData_SDA_AH_group2.csv'
Y_LABEL = 'Z_OppositeTeamDefence'
PREDICTORS_NUMBER = 8


class UtilsBayes(utils.Utils):
    def create_classifier(self):
        return GaussianNB()


Class = (utils.Utils, UtilsBayes)[0]  # 0 = Logistic regression, 1 = GaussianNB

utils_ = Class(ABS_PATH, DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)
df = utils_.read_dataset()

X, y = utils_.getXy(df)

# interaction_column = X['Y_AvgTravelledDistance'] * X['Y_MentalConcentration']
# XCattiva = pd.concat([X, interaction_column.to_frame()], axis=1)
# XCattiva = interaction_column.to_frame()
# model = utils_.train_separate_df(X, y)
# score = utils_.inspect_model(model, X, y, n=100, k=5)
# print('CV accuracy: ', score)

# print('Deviance Marco: ', 2 * metrics.log_loss(y,
#       model.predict_proba(X), normalize=False))

# formula = ' + '.join(['Z_OppositeTeamDefence ~ Y_AvgTravelledDistance*Y_MentalConcentration', 'Y_Dehydration', 'Y_Hyperthermia', 'Y_AvgSpeed', 'Y_AvgTravelledDistance',
#                       'Y_PressingCapability', 'Y_PhysicalEndurance', 'Y_MentalConcentration', 'Y_EmotionalMotivation'])
# print(formula)
# logitModel = sm.Logit(y, sm.add_constant(XCattiva)).fit()

# print('Deviance Utils: ', sm.Logit(y, XCattiva).fit().deviance)

# score = model.score(XCattiva, y)
# print("Training set accuracy: ", score)

plt.figure(figsize=(12, 5), dpi=80)
utils_.plot_heatmap(df)
