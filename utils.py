from sklearn.linear_model import LogisticRegression
import pandas as pd
import os
from sklearn.model_selection import cross_validate
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import log_loss
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


class Utils:
    def __init__(self, path, filename, y_label, predictors_number):
        self.path = path
        self.filename = filename
        self.y_label = y_label
        self.predictors_number = predictors_number

    def read_dataset(self):
        return pd.read_csv(os.path.join(self.path, self.filename))

    def logisticRegression(self, df: pd.DataFrame, return_values=False):
        X, y = self.getXy(df)
        return self.logisticRegressionSeparateDf(X, y, return_values)

    def logisticRegressionSeparateDf(self, X: pd.DataFrame, y: pd.DataFrame, return_values=False):
        if not return_values:
            return LogisticRegression().fit(X, y)
        return LogisticRegression().fit(X, y), X, y

    def to_numpy(self, df):
        return self.getXy(df, True)

    def getXy(self, df, to_numpy=False):
        X, y = None, None
        X = df.loc[:, df.columns != self.y_label]
        y = df[self.y_label]
        if(to_numpy):
            X = X.to_numpy()
            y = y.to_numpy()
        return X, y

    def inspectLr(self, model, X, y, n=1000, k=5):
        scores = cross_validate(model, X, y, n_jobs=5, cv=k)
        fpr, tpr, thresholds = plot_roc_curve(model, X, y)
        plt.show()
        return sum(scores["test_score"])/len(scores["test_score"])

    def pretty(self, d, indent=0):
        for key, value in d.items():
            print('\t' * indent + str(key))
            if isinstance(value, dict):
                self.pretty(value, indent+1)
            else:
                print('\t' * (indent+1) + str(value))

    def deviance(self, X, y, model):
        return 2*log_loss(y, model.predict_log_proba(X))

    def generateInteractionValues(self, df, return_names=False, add_squares=False):
        X, _ = self.getXy(df)
        xlabels = list(X.columns)

        interactions = None
        interactions = set(tuple(sorted((xlabel, ylabel)))
                           for xlabel in xlabels for ylabel in xlabels if xlabel != ylabel or add_squares)
        if(return_names):
            return ((df[xlabel] * df[ylabel], xlabel, ylabel) for xlabel, ylabel in interactions)
        return (df[xlabel] * df[ylabel] for xlabel, ylabel in interactions)

    def plot_heatmap(self, df):
        sns.set_theme()

        X, y = self.getXy(df)
        xlabels = X.columns

        mat = np.zeros((len(xlabels), len(xlabels)))

        for interaction, xlabel, ylabel in self.generateInteractionValues(df, return_names=True, add_squares=True):
            X_with_interaction = pd.concat([df[xlabels], interaction], axis=1)
            lr = self.logisticRegressionSeparateDf(X_with_interaction, y)
            dev = self.deviance(X_with_interaction, y, lr)

            xindex = xlabels.index(xlabel)
            yindex = xlabels.indexof(ylabel)
            mat[xindex, yindex] = dev

        return sns.heatmap(
            mat, annot=True, fmt="d", xticklabels=xlabels, yticklabels=xlabels)
