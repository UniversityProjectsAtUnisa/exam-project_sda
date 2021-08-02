from sklearn.linear_model import LogisticRegression
import pandas as pd
import os
from sklearn.model_selection import cross_validate
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import log_loss
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
from sklearn.naive_bayes import GaussianNB
from statsmodels import formula
import statsmodels.api as sm
from itertools import chain, combinations
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import cProfile
import pstats
import io
import pickle


CLASSIFICATION_MODEL_NAME = "Classification.pickle"


def profile(fnc):
    """A decorator that uses cProfile to profile a function"""
    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


class Utils:
    def __init__(self, path, filename, y_label, predictors_number, gaussian_classifier=False):
        self.path = path
        self.filename = filename
        self.y_label = y_label
        self.predictors_number = predictors_number
        self.gaussian_classifier = gaussian_classifier

    def create_classifier(self):
        if self.gaussian_classifier:
            return GaussianNB()
        return LogisticRegression(max_iter=100000, penalty='none')

    def read_dataset(self):
        return pd.read_csv(os.path.join(self.path, self.filename))

    def train(self, df: pd.DataFrame, return_values=False):
        X, y = self.getXy(df)
        return self.train_separate_df(X, y, return_values)

    def train_separate_df(self, X: pd.DataFrame, y: pd.DataFrame, return_values=False):
        if not return_values:
            return self.create_classifier().fit(X, y)
        return self.create_classifier().fit(X, y), X, y

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

    def inspect_model(self, model, X, y, n=100, k=5):
        total_score = 0
        for i in range(n):
            kf = StratifiedKFold(n_splits=k, random_state=i, shuffle=True)
            scores = cross_validate(model, X, y, cv=kf)
            total_score += np.mean(scores["test_score"])

        return total_score / n

    def pretty(self, d, indent=0):
        for key, value in d.items():
            print('\t' * indent + str(key))
            if isinstance(value, dict):
                self.pretty(value, indent+1)
            else:
                print('\t' * (indent+1) + str(value))

    # def deviance(self, df, formula):
    def comparison_score(self, X, y, model):
        if self.gaussian_classifier:
            return model.score(X, y)
        return 2 * metrics.log_loss(y, model.predict_proba(X), normalize=False)

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
        X, y = self.getXy(df)
        xlabels = list(X.columns)

        d = defaultdict(dict)

        for interaction, xlabel, ylabel in self.generateInteractionValues(df, return_names=True, add_squares=True):

            interaction = interaction.to_frame()
            interaction = interaction.rename(columns={0: xlabel+'*'+ylabel})

            X_with_interaction = pd.concat([df[xlabels], interaction], names=(
                xlabels + list(interaction.columns)), axis=1)

            model = self.train_separate_df(X_with_interaction, y)

            d[ylabel][xlabel] = self.comparison_score(
                X_with_interaction, y, model)

        data = pd.DataFrame(d)
        return sns.heatmap(data.sort_index(axis=0)[sorted(data.columns)],
                           fmt='.5f', annot=True, cmap="YlGnBu",
                           cbar_kws={'label': 'Accuracy'if self.gaussian_classifier else 'Deviance'})

    def _powerset(self, iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    def best_subset(self, df: pd.DataFrame, possible_interactions, nfolds=5, nCV=5, verbose=True):
        n_predictors = len(possible_interactions) + self.predictors_number
        X, _ = self.getXy(df)
        xlabels = list(X.columns) + [x+'*'+y for x, y in possible_interactions]

        # create dataset with all interactions
        df_interactions = df.copy()
        for x, y in possible_interactions:
            inter = df_interactions[x] * df_interactions[y]
            inter = pd.DataFrame({(x+'*'+y): inter})
            df_interactions = pd.concat([df_interactions, inter], axis=1)

        results = {
            'best_models': {},
            'accuracies': {},
            'comparison_scores': {},
        }

        _, y = self.getXy(df)

        for subset in tqdm(self._powerset(list(range(0, n_predictors))), total=2**n_predictors):
            if (len(subset) == 0):
                continue
            # predictors used in the current iteration
            selected_predictors = set(xlabels[i] for i in subset)

            # add main effects in selected predictors
            for x_, y_ in (xy.split('*') for xy in list(selected_predictors) if '*' in xy):
                selected_predictors.add(x_)
                selected_predictors.add(y_)

            X_with_interactions = df_interactions[selected_predictors]
            model = self.train_separate_df(X_with_interactions, y)
            formula = self.y_label + '~' + '+'.join(selected_predictors)
            num_predictors_in_subset = len(selected_predictors)

            self._assert_main_effects(formula, num_predictors_in_subset)
            comparison_score = self.comparison_score(X_with_interactions, y, model)

            if (num_predictors_in_subset not in results['comparison_scores'] or
                    results['comparison_scores'][num_predictors_in_subset] < comparison_score):
                results['best_models'][num_predictors_in_subset] = {
                    'model': model, 'formula': formula, 'X': X_with_interactions, 'y': y}
                results['comparison_scores'][num_predictors_in_subset] = comparison_score

        for predictors_number in results['best_models'].keys():
            model_info = results['best_models'][predictors_number]
            accuracy = self.inspect_model(
                model_info['model'], model_info['X'], model_info['y'], k=nfolds, n=nCV)
            results['accuracies'][predictors_number] = accuracy

        return results

    @classmethod
    def plot_best_subsets(cls, best_subsets, n_predictors, ax, color=None, legend=None):
        x = np.arange(1, n_predictors+1)
        ax.plot(x, [best_subsets['accuracies'][i]
                 for i in x], 'o--', c=color if color is not None else 'orange', label = legend)
        if legend:
            ax.legend()
        ax.set_title('Best accuracies vs predictors')
        ax.grid(b=True,which='major')
        ax.set_xlabel('Number of predictors')
        ax.set_ylabel('Accuracy')

    def _assert_main_effects(self, formula, num_predictors_in_subset):
        # print('assert:',formula)
        pieces = set(formula[(formula.index('~') + 1):].split('+'))
        for piece in pieces:
            if '*' in piece:
                x, y = piece.split('*')
                assert((x in pieces) and (y in pieces))

    def save_model(self, model):
        with open(os.path.join(self.path, CLASSIFICATION_MODEL_NAME), 'wb') as f:
            pickle.dump(model, f)
