from utils import CLASSIFICATION_MODEL_NAME
import pandas as pd
import os
from collections import defaultdict
import pickle


ABS_PATH = r"C:\Users\carbo\OneDrive\Documenti\Magistrale Carbone\2 sem\Statistical Data Analysis\aPROGETTO\SDAgruppo2"
ABS_PATH = '/mnt/c/Users/marco/Documents/UNISA/SDA/progetto/SDAgruppo2'
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

Y_LABELS = [
    "Y_PressingCapability",
]


def read_regression_coefficients(path):
    Y_COEFFICIENTS = defaultdict(dict)

    for label in Y_LABELS:
        res = pd.read_csv(os.path.join(path, f'{label}.csv')).to_dict()
        names = res['Unnamed: 0']
        coeffs = res['model.coefficients']
        Y_COEFFICIENTS[label] = {names[i]: coeffs[i] for i in names}
        print(Y_COEFFICIENTS[label])

    return Y_COEFFICIENTS


def read_classification_model(path, filename):
    return pickle.load(os.path.join(path, filename))


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
    y = [] # LIST OF DICTIONARIES
    for _, row in df.iterrows():
        y_row = calculate_y(row, coefficients)
        y.append(y_row)

    return y


def main():
    df = read_dataset(ABS_PATH, DATASET_FILENAME)

    coefficients = read_regression_coefficients(ABS_PATH)

    y = predict_y(df, coefficients)

    y_table = y_to_df(y)

    model = read_classification_model(ABS_PATH, CLASSIFICATION_MODEL_NAME)

    z = model.predict(y_table)

    # z = z.to_data_frame()

    print(z)
    z.to_csv(os.path.join(ABS_PATH, OUTPUT_FILE))


if __name__ == "__main__":
    main()
