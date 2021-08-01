from utils import CLASSIFICATION_MODEL_NAME
import pandas as pd
import os
from collections import defaultdict
import pickle
import re


ABS_PATH = r"C:\Users\carbo\OneDrive\Documenti\Magistrale Carbone\2 sem\Statistical Data Analysis\aPROGETTO\SDAgruppo2"
ABS_PATH = r'C:\Users\gorra\Desktop\gitSDA\SDAgruppo2'
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
    Y_COEFFICIENTS = defaultdict(dict)

    for label in Y_LABELS:
        res = pd.read_csv(os.path.join(path, f'{label}.csv')).to_dict()
        names = res['Unnamed: 0']
        coeffs = res['model.coefficients']
        Y_COEFFICIENTS[label] = {names[i]: coeffs[i] for i in names}

    return Y_COEFFICIENTS


def read_classification_model(path, filename):
    with open(os.path.join(path, filename),'rb') as f:
        return pickle.load(f)


def read_dataset(path, filename):
    return pd.read_csv(os.path.join(path, filename))


def calculate_y(row, coefficients):
    predictions = {}
    for label, coef_of_label in coefficients.items():
        prediction = 0
        for coef_name, betaj in coef_of_label.items():
            if coef_name == "(Intercept)":
                prediction += betaj
            else:
                prediction += betaj * row[coef_name]

        predictions[label] = prediction
    return predictions


def y_to_df(y):
    return pd.DataFrame(y)


def predict_y(df, coefficients):
    y = []  # LIST OF DICTIONARIES
    for _, row in df.iterrows():
        y_row = calculate_y(row, coefficients)
        y.append(y_row)

    return y


def mysqrt(x):
    if(x >= 0):
        return((x)**0.5)
    else:
        return -((-x)**0.5)


def solve_interaction(df, name):
    if ":" in name:
        predictor1, predictor2 = name.split(':')[0], name.split(':')[1]
        return solve_interaction(df,predictor1)*solve_interaction(df,predictor2)
    m = re.search('.*I\\((.*)\\^(\\d+)\\)', name)
    if m is not None:
        predictor = m.group(1)
        power = m.group(2)
        squared = (df[predictor])**float(power)
        return squared

    elif "map_dbl" in name:
        print('mapdbl,',name)
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
        if ('Intercept' in name) : continue
        print(name)
        expanded_df[name] = solve_interaction(df, name)
    return expanded_df



def main():
    df = read_dataset(ABS_PATH, DATASET_FILENAME)

    coefficients = read_regression_coefficients(ABS_PATH)


    coef_names = set()
    for _, coef_of_label in coefficients.items():
        for coef_name in coef_of_label:
            coef_names.add(coef_name)

    df = expand_df(df, coef_names)

    y = predict_y(df, coefficients)

    y_table = y_to_df(y)

    model = read_classification_model(ABS_PATH, CLASSIFICATION_MODEL_NAME)
    
    formula = model['formula']
    coef_names = list(formula.split('~')[1].split('+'))   

    y_table_expanded = expand_df(y_table, coef_names)
    print("coef_names",coef_names)
    print(y_table_expanded)

    z = model['model'].predict(y_table_expanded[coef_names])
    print(z)
    exit(0)

    print(z)
    pd.DataFrame({'Class':z}).to_csv(os.path.join(ABS_PATH, OUTPUT_FILE))


if __name__ == "__main__":
    main()
