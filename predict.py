import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from utils import CLASSIFICATION_MODEL_NAME
import pandas as pd
import os
from collections import defaultdict
import pickle
import re


ABS_PATH = r"C:\Users\carbo\OneDrive\Documenti\Magistrale Carbone\2 sem\Statistical Data Analysis\aPROGETTO\SDAgruppo2"
ABS_PATH = r'C:\Users\gorra\Desktop\gitSDA\SDAgruppo2'
ABS_PATH = '/home/marco741/SDAgruppo2'
DATASET_FILENAME = 'FINAL.csv'
TRAINING_DATASET_FILENAME = 'RegressionData_SDA_AH_group2.csv'
X_LABELS = []
PREDICTORS_NUMBER = 10
CLASSIFICATION_MODEL_NAME = "Classification.pickle"
OUTPUT_FILE = 'ANSWER.csv'

Y_LABELS = [
    "Y_Dehydration",
    "Y_Hyperthermia",
    "Y_AvgSpeed",
    "Y_AvgTravelledDistance",
    "Y_PressingCapability",
    "Y_PhysicalEndurance",
    "Y_MentalConcentration",
    "Y_EmotionalMotivation",
]


def read_regression_models(path):
    """
    Returns:
        {
            "Y_MentalConcentration": {
                "coefficients": {
                    "(Intercept)": value,
                    "X_Temperature": value,
                    ...
                }, 
                "is_GLM": True
            },
            ...
            "Y_PressingCapability": {
                "coefficients": {
                    "(Intercept)": value,
                    "X_Temperature": value,
                    ...
                }, 
                "is_GLM": False
            },
        }
    """
    models_info = {}

    for label in Y_LABELS:
        res = pd.read_csv(os.path.join(path, f'{label}.csv')).to_dict()

        is_GLM = 'is_GLM' in res

        names = res['is_GLM' if is_GLM else 'not_GLM']
        coeffs = res['model.coefficents']
        coefficients_info = {(names[i]).replace('*', ":"): coeffs[i]
                             for i in names if names[i] != 'is_GLM'}
        models_info[label] = {
            'coefficients': coefficients_info, 'is_GLM': is_GLM}

    return models_info


def read_classification_model(path, filename):
    with open(os.path.join(path, filename), 'rb') as f:
        return pickle.load(f)


def read_dataset(path, filename):
    return pd.read_csv(os.path.join(path, filename))


def y_to_df(y):
    return pd.DataFrame(
        {ylabel: x.reshape(-1) for ylabel, x in y.items()})


def scale(df, coef_names):
    training_df = read_dataset(ABS_PATH, TRAINING_DATASET_FILENAME)
    training_df = expand_df(training_df, coef_names)
    scaled_df = pd.DataFrame()

    scaled_df = (df-training_df.mean())/training_df.std()

    return scaled_df


def predict_y(df, models_info, coef_names):
    y = {}

    scaled_df = scale(df, coef_names)

    for label in models_info:
        model = LinearRegression()
        model.coef_ = np.zeros(len(df.columns))
        for idx, predictor in enumerate(df.columns):
            model.coef_[idx] = models_info[label]['coefficients'].get(
                predictor, 0)
        model.coef_ = model.coef_.reshape(1, -1)
        model.intercept_ = np.array(
            [models_info[label]['coefficients']["(Intercept)"]])

        if(models_info[label]['is_GLM']):
            y[label] = model.predict(scaled_df)
        else:
            y[label] = model.predict(df)

    return y


def mysqrt(x):
    if(x >= 0):
        return((x)**0.5)
    else:
        return -((-x)**0.5)


def solve_interaction(df, name):
    if ":" in name:
        predictor1, predictor2 = name.split(':')[0], name.split(':')[1]
        return solve_interaction(df, predictor1)*solve_interaction(df, predictor2)
    m = re.search('.*I\\((.*)\\^(\\d+)\\)', name)
    if m is not None:
        predictor = m.group(1)
        power = m.group(2)
        squared = (df[predictor])**float(power)
        return squared

    elif "map_dbl" in name:
        between = name[name.index('(')+1: name.index(')')]
        predictor, funct = between.split(',')
        predictor = predictor.strip()
        funct = funct.strip()
        newcolumn = df[predictor].apply(eval(funct))
        return newcolumn
    return df[name]


def expand_df(df, coef_names):
    # Aggiungi le colonne a df
    expanded_df = pd.DataFrame()
    for name in coef_names:
        if 'Intercept' in name:
            continue
        expanded_df[name] = solve_interaction(df, name)
    return expanded_df


def get_coef_names(models_info):
    coef_names = set()

    for label in models_info:
        for coef_name in models_info[label]['coefficients']:
            coef_names.add(coef_name.replace('*', ':'))
    return coef_names


def main():
    print("Reading final dataset")
    df = read_dataset(ABS_PATH, DATASET_FILENAME)

    print("Retrievivng coefficients from best models previously trained")
    models_info = read_regression_models(ABS_PATH)

    print('Predicting the intermediate results')
    coef_names = get_coef_names(models_info)
    df = expand_df(df, coef_names)
    y = predict_y(df, models_info, coef_names)
    y_table = y_to_df(y)

    print('Retrieving the best classification model previously trained')
    model = read_classification_model(ABS_PATH, CLASSIFICATION_MODEL_NAME)

    print('Predicting the final results')
    formula = model['formula']
    coef_names = list(formula.replace('*', ':').split('~')[1].split('+'))
    y_table_expanded = expand_df(y_table, coef_names)
    z = model['model'].predict(y_table_expanded[coef_names])
    print('Printing and save output')
    pd.DataFrame({'Class': z}).to_csv(os.path.join(ABS_PATH, OUTPUT_FILE))
    print(z)

    # # ### fake
    # X = np.array([1]+list(range(11)))
    # X =np.array([1]+[16.6158265 , 55.40407603, 13.28890599,  3.15280286, 41.89730087,

    #         3.14540185, -0.21817317, 10.48807792,  6.43804884, 11.87052228,
    #         5.2825767 ])
    # beta = np.array([model['model'].intercept_] + model['model'].coef_.reshape(-1).tolist()).reshape(-1)
    # yhat = X @ beta
    # def sigmoid(x):
    #     return np.exp(x)/(1+np.exp(x))

    # yhat = np.array([sigmoid(x) for x in (yhat).reshape(-1)])

    # classes = model['model'].predict(X[1:].reshape(1,-1))

    # print(classes)
    # a = np.array([1 if a_ > 0.5 else 0 for a_ in yhat])
    # print('ifinale, ',sum(a-classes))



if __name__ == "__main__":
    main()
