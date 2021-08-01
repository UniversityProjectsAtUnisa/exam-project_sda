import numpy as np
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


def read_regression_coefficients(path):
    """
    Returns:
        {
            "Y_MentalConcentration": {
                "(Intercept)": value,
                "X_Temperature": value,
                ...
            },
            ...
            "Y_PressingCapability": {
                "(Intercept)": value,
                "X_Temperature": value,
                ...
            },
        }
    """
    y_coefficients = defaultdict(dict)

    for label in Y_LABELS:
        res = pd.read_csv(os.path.join(path, f'{label}.csv')).to_dict()
        names = res['Unnamed: 0']
        coeffs = res['model.coefficients']
        y_coefficients[label] = {names[i]: coeffs[i] for i in names}

    return y_coefficients


def read_classification_model(path, filename):
    with open(os.path.join(path, filename), 'rb') as f:
        return pickle.load(f)


def read_dataset(path, filename):
    return pd.read_csv(os.path.join(path, filename))


def y_to_df(y):
    return pd.DataFrame(
        {ylabel: x.reshape(-1) for ylabel, x in y.items()})


def predict_y(df, coefficients):
    y = {}
    for y_name, coeffs_of_y_name in coefficients.items():
        model = LinearRegression()
        model.coef_ = np.zeros(len(df.columns))
        for idx, predictor in enumerate(df.columns):
            model.coef_[idx] = coeffs_of_y_name.get(predictor, 0)

        model.coef_ = model.coef_.reshape(1, -1)
        model.intercept_ = np.array([coeffs_of_y_name["(Intercept)"]])
        y[y_name] = model.predict(df)
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
        if ('Intercept' in name):
            continue
        expanded_df[name] = solve_interaction(df, name)
    return expanded_df


def get_coef_names(coefficients):
    coef_names = set()
    for _, coef_of_label in coefficients.items():
        for coef_name in coef_of_label:
            coef_names.add(coef_name)
    return coef_names


def main():
    print("Reading final dataset")
    df = read_dataset(ABS_PATH, DATASET_FILENAME)

    print("Retrievivng coefficients from best models previously trained")
    coefficients = read_regression_coefficients(ABS_PATH)

    print('Predicting the intermediate results')
    df = expand_df(df, get_coef_names(coefficients))
    y = predict_y(df, coefficients)
    y_table = y_to_df(y)

    print('Retrieving the best classification model previously trained')
    model = read_classification_model(ABS_PATH, CLASSIFICATION_MODEL_NAME)

    print('Predicting the final results')
    formula = model['formula']
    coef_names = list(formula.split('~')[1].split('+'))
    y_table_expanded = expand_df(y_table, coef_names)
    z = model['model'].predict(y_table_expanded[coef_names])

    print('Printing and save output')
    pd.DataFrame({'Class': z}).to_csv(os.path.join(ABS_PATH, OUTPUT_FILE))
    print(z)
    exit(0)


if __name__ == "__main__":
    main()
