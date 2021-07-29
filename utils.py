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
import statsmodels.api as sm
from itertools import chain, combinations
from tqdm import tqdm

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
            return LogisticRegression(max_iter=1000).fit(X, y)
        return LogisticRegression(max_iter=1000).fit(X, y), X, y

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
        ################ NOTA: da fare la media, non ci scordiamo!!!!!!!!!!! ##############àààà
        scores = cross_validate(model, X, y, n_jobs=5, cv=k)
        #fpr, tpr, thresholds = plot_roc_curve(model, X, y)
        #plt.show()
        return sum(scores["test_score"])/len(scores["test_score"])

    def pretty(self, d, indent=0):
        for key, value in d.items():
            print('\t' * indent + str(key))
            if isinstance(value, dict):
                self.pretty(value, indent+1)
            else:
                print('\t' * (indent+1) + str(value))

    #def deviance(self, X, y, model):
    def deviance(self, df, formula):

        fit = sm.GLM.from_formula(formula, data=df, family=sm.families.Binomial()).fit()
        return fit.deviance

        z=model.predict_log_proba(X)

        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                z[i,j] = max(z[i,j], -1e+150)
                z[i,j] = min(-1e-40, z[i,j])
                if (z[i,j]==0) :print('problema ooooooooooooo',z[i,j])
                
        print('fino a qua tutto appost')
        try:
            return 2*log_loss(y, z)
        except  Exception as ex:
            print('errore su questi',ex)
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
        sns.set_theme()

        X, y = self.getXy(df)
        xlabels = list(X.columns)

        d = defaultdict(dict)
        # NOTA: rispettare il principio gerarchico
        for interaction, xlabel, ylabel in self.generateInteractionValues(df, return_names=True, add_squares=True):
            if ('Y_AvgTravelledDistance*Y_Hyperthermia'==(xlabel+'*'+ylabel)):
                continue
            #print(xlabel, ylabel, 'start')
            #print('interaction','\n',interaction)
            interaction=interaction.to_frame()
            interaction=interaction.rename(columns={"0": xlabel+'*'+ylabel})
            X_with_interaction = pd.concat([df[xlabels], interaction], axis=1)
            try:
                lr = self.logisticRegressionSeparateDf(X_with_interaction, y)
            except Exception as e :
                print('errore qua')
                print(e)
                print(X_with_interaction)
                exit(0)
            #xindex = xlabels.index(xlabel)
            #yindex = xlabels.index(ylabel)

            #dev = xlabel+'*'+ylabel#self.deviance(X_with_interaction, y, lr)
            #dev=self.deviance(X_with_interaction, y, lr)

            formula = self.y_label + '~'+ '+'.join(xlabels + [xlabel+'*'+ylabel])
            #print('formula',formula)
            dev=self.deviance(pd.concat([X_with_interaction, df[[self.y_label]]], axis=1), formula)
            d[ylabel][xlabel] = dev#xindex*10+yindex


        return sns.heatmap(pd.DataFrame(d),fmt='.5f', annot=True, cmap="YlGnBu")


    def powerset(self, iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    def best_subset(self, df, possible_interactions, nfolds = 5, nCV = 5, verbose=True):

        n_predictors = len(possible_interactions) + len(df.columns) - 1
        X, _ = self.getXy(df)
        xlabels = list(X.columns) + [x+'*'+y for x,y in possible_interactions]

        #create dataset with all interactions
        df_interactions = df.copy()
        for x,y in possible_interactions:
            inter = (df_interactions[x] * df_interactions[y])
            inter= pd.DataFrame({(x+'*'+y):inter})
            df_interactions = pd.concat([df_interactions, inter], axis=1)

        print('itnerazioni aggiunte', list(df_interactions.columns))

        results = {
            'best_models': {},
            'accuracies' : {},
            'formulas'   : {}
        }

        _, y = self.getXy(df)
        for subset in tqdm(self.powerset(list(range(0,n_predictors))), total = 2**n_predictors):
            if (len(subset) == 0): continue
            selected_predictors = [xlabels[i] for i in subset]

            X_with_interactions = df_interactions[selected_predictors]
            lr = self.logisticRegressionSeparateDf(X_with_interactions, y)
            accuracy = self.inspectLr(lr, X_with_interactions, y, k=nfolds)
            
            num_predictors_in_subset = len(subset)

            if (num_predictors_in_subset not in results['accuracies'] or
                 results['accuracies'][num_predictors_in_subset] < accuracy): #### NOTA: da cambiare con la devianza
                results['best_models'][num_predictors_in_subset] = lr
                results['accuracies'][num_predictors_in_subset] = accuracy
                results['formulas'][num_predictors_in_subset] = selected_predictors


        x = np.arange(1, n_predictors+1)
        plt.plot(x, [results['accuracies'][i] for i in x], 'o--', c='orange')
        plt.title('Best accuracies vs predictors')
        plt.grid()
        plt.xlabel('Number of predictors')
        plt.ylabel('Accuracy')
        return results