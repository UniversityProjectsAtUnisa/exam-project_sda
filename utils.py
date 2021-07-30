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
from statsmodels import formula
import statsmodels.api as sm
from itertools import chain, combinations
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold


class Utils:
    def __init__(self, path, filename, y_label, predictors_number):
        self.path = path
        self.filename = filename
        self.y_label = y_label
        self.predictors_number = predictors_number

    def create_classifier(self):
        return LogisticRegression(max_iter=1000)

    def read_dataset(self):
        return pd.read_csv(os.path.join(self.path, self.filename))

    def logisticRegression(self, df: pd.DataFrame, return_values=False):
        X, y = self.getXy(df)
        return self.logisticRegressionSeparateDf(X, y, return_values)

    def logisticRegressionSeparateDf(self, X: pd.DataFrame, y: pd.DataFrame, return_values=False):
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

    def inspectLr(self, model, X, y, n=100, k=5):
        # NOTA: da fare la media, non ci scordiamo!!!!!!!!!!! ##############àààà
        total_score = 0
        for i in range(n):
            kf = StratifiedKFold(n_splits=k, random_state=i, shuffle=True)
            scores = cross_validate(model, X, y, n_jobs=5, cv=kf)
            # print(scores['test_score'])
            total_score += np.mean(scores["test_score"])

        total_score = total_score / n
        # print(total_score)
        # assert(False)
        return total_score

    def pretty(self, d, indent=0):
        for key, value in d.items():
            print('\t' * indent + str(key))
            if isinstance(value, dict):
                self.pretty(value, indent+1)
            else:
                print('\t' * (indent+1) + str(value))

    # def deviance(self, X, y, model):
    def deviance(self, df, formula):

        fit = sm.GLM.from_formula(
            formula, data=df, family=sm.families.Binomial()).fit()
        return fit.deviance

        z = model.predict_log_proba(X)

        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                z[i, j] = max(z[i, j], -1e+150)
                z[i, j] = min(-1e-40, z[i, j])
                if (z[i, j] == 0):
                    print('problema ooooooooooooo', z[i, j])

        print('fino a qua tutto appost')
        try:
            return 2*log_loss(y, z)
        except Exception as ex:
            print('errore su questi', ex)
            print(y, z)

            exit(0)

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
        print('computing interactions...')
        sns.set_theme()

        X, y = self.getXy(df)
        xlabels = list(X.columns)

        d = defaultdict(dict)

        for interaction, xlabel, ylabel in self.generateInteractionValues(df, return_names=True, add_squares=True):
            if ('Y_AvgTravelledDistance*Y_Hyperthermia' == (xlabel+'*'+ylabel)):
                continue
            #print(xlabel, ylabel, 'start')
            # print('interaction','\n',interaction)
            interaction = interaction.to_frame()
            interaction = interaction.rename(columns={"0": xlabel+'*'+ylabel})
            X_with_interaction = pd.concat([df[xlabels], interaction], axis=1)
            try:
                lr = self.logisticRegressionSeparateDf(X_with_interaction, y)
            except Exception as e:
                print('errore qua')
                print(e)
                print(X_with_interaction)
                exit(0)
            #xindex = xlabels.index(xlabel)
            #yindex = xlabels.index(ylabel)

            # dev = xlabel+'*'+ylabel#self.deviance(X_with_interaction, y, lr)
            #dev=self.deviance(X_with_interaction, y, lr)

            formula = self.y_label + '~' + \
                '+'.join(xlabels + [xlabel+'*'+ylabel])
            # print('formula',formula)
            dev = self.deviance(
                pd.concat([X_with_interaction, df[[self.y_label]]], axis=1), formula)
            d[ylabel][xlabel] = dev  # xindex*10+yindex

        data = pd.DataFrame(d)
        return sns.heatmap(data.sort_index(axis=0)[sorted(data.columns)],
                           fmt='.5f', annot=True, cmap="YlGnBu",
                           cbar_kws={'label': 'Deviance'})

    def _powerset(self, iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    def best_subset(self, df, possible_interactions, nfolds=5, nCV=5, verbose=True):

        n_predictors = len(possible_interactions) + len(df.columns) - 1
        X, _ = self.getXy(df)
        xlabels = list(X.columns) + [x+'*'+y for x, y in possible_interactions]

        # create dataset with all interactions
        df_interactions = df.copy()
        for x, y in possible_interactions:
            inter = (df_interactions[x] * df_interactions[y])
            inter = pd.DataFrame({(x+'*'+y): inter})
            df_interactions = pd.concat([df_interactions, inter], axis=1)

        # print('itnerazioni aggiunte', list(df_interactions.columns))

        results = {
            'best_models': {},
            'accuracies': {},
            'deviances': {},
            'perfect separation': {}
        }

        _, y = self.getXy(df)

        # NOTA: evitare di calcolare le perfect separation formulas
        perfect_separation_formulas = {
            'Z_OppositeTeamDefence~Y_Dehydration+Y_PhysicalEndurance+Y_MentalConcentration+Y_Hyperthermia+Y_EmotionalMotivation+Y_AvgTravelledDistance+Y_AvgTravelledDistance*Y_PhysicalEndurance+Y_PressingCapability+Y_EmotionalMotivation*Y_PhysicalEndurance+Y_AvgSpeed',
            'Z_OppositeTeamDefence~Y_Dehydration+Y_PhysicalEndurance+Y_MentalConcentration+Y_Hyperthermia+Y_EmotionalMotivation+Y_AvgTravelledDistance+Y_AvgTravelledDistance*Y_PhysicalEndurance+Y_EmotionalMotivation*Y_PhysicalEndurance+Y_AvgSpeed',
            'Z_OppositeTeamDefence~Y_Dehydration+Y_PhysicalEndurance+Y_MentalConcentration+Y_EmotionalMotivation+Y_AvgTravelledDistance+Y_AvgTravelledDistance*Y_PhysicalEndurance+Y_PressingCapability+Y_EmotionalMotivation*Y_PhysicalEndurance+Y_AvgSpeed'
        }

        # NOTA: rispettare il principio gerarchico
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
            lr = self.logisticRegressionSeparateDf(X_with_interactions, y)
            accuracy = self.inspectLr(
                lr, X_with_interactions, y, n=nCV, k=nfolds)
            formula = self.y_label + '~' + '+'.join(selected_predictors)
            num_predictors_in_subset = len(selected_predictors)

            self._assert_main_effects(formula, num_predictors_in_subset)
            try:
                if formula not in perfect_separation_formulas:
                    deviance = self.deviance(df, formula=formula)
                else:
                    deviance = 0
            except Exception as ex:
                print(ex)
                print('fromula:', formula)
                if ('Perfect separation detected, results not available' in str(ex)):
                    perfect_separation_formulas.add(formula)
                else:
                    exit(0)
            # print('formula in current iteration:',
            #   (self.y_label + '~' + '+'.join(selected_predictors)))

            #print('num predictors =', num_predictors_in_subset)

            if (num_predictors_in_subset not in results['deviances'] or
                    results['deviances'][num_predictors_in_subset] < deviance):  # NOTA: da cambiare con la devianza
                results['best_models'][num_predictors_in_subset] = {
                    'model': lr, 'formula': formula}
                results['accuracies'][num_predictors_in_subset] = accuracy
                results['perfect separation'][num_predictors_in_subset] = (
                    formula in perfect_separation_formulas)
                results['deviances'][num_predictors_in_subset] = deviance

        x = np.arange(1, n_predictors+1)
        plt.plot(x, [results['accuracies'][i] for i in x], 'o--', c='orange')
        plt.title('Best accuracies vs predictors')
        plt.grid()
        plt.xlabel('Number of predictors')
        plt.ylabel('Accuracy')
        return results, perfect_separation_formulas

    def _assert_main_effects(self, formula, num_predictors_in_subset):
        # print('assert:',formula)
        pieces = set(formula[(formula.index('~') + 1):].split('+'))
        for piece in pieces:
            if '*' in piece:
                x, y = piece.split('*')
                assert((x in pieces) and (y in pieces))
