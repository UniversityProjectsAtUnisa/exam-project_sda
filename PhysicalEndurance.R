#==============================   CONFIG     ============================

ABS_PATH = 'C:/Users/gorra/Desktop/new_git/SDAgruppo2'
DATASET_FILENAME = 'RegressionData_SDA_AH_group2.csv'
Y_LABEL = 'Y_PhysicalEndurance'
PREDICTORS_NUMBER = 10

#================================ START =================================

setwd(ABS_PATH)
source('./utils.R')
ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)


#==================== REGRESSION WITHOUT INTERACTIONS ====================

bestSubsets = bestSubsetSelection(ds, relationships=NULL, nMSE=10, folds=5, verbose=T, method="exhaustive")
ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

bestSubset = bestSubsets$model[[4]]

lm.inspect(bestSubset, 10, 10)
# MSE = 3.85


#====================== INSPECT RELATIONSHIPS ============================

showPlotsAgainstOutput(ds, 2:(PREDICTORS_NUMBER+1))



#======================== TEST RELATIONSHIPS =============================

possibleDependencies = list('X_Temperature',
                            'exp(-X_RestTimeFromLastMatch)',
                            'X_Humidity',
                            'X_SupportersImpact'
)

dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(dependencyModel, 5)
# RSquared: 0.7763, MSE: 3.6125

#########################################################################

possibleDependencies = list('X_Temperature',
                            'exp(-X_RestTimeFromLastMatch)',
                            'X_Humidity',
                            'X_SupportersImpact',
                            'X_MatchRelevance'
)

dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(dependencyModel, 5)
# RSquared: 0.8041, MSE: 3.459765    # BEST
# inoltre i residui migliorano notevolmente 


#======================== INSPECT INTERACTIONS =============================

# Collect rsquared for every linear model obtained by adding every possible
# interaction between two distinct predictors to the base model.
# Set base rsquared as default value

baseRSquared = summary( lm.byIndices(ds, -1) )$r.squared
interactionMatrix = inspectInteractionMatrix(ds, default=baseRSquared, showHeatmap = T)



#========================  TEST INTERACTIONS   =============================

possibleDependencies = list('I(X_Temperature^2)',
                            'X_Humidity',
                            'X_SupportersImpact',
                            'X_MatchRelevance'
)

possibleInteractions = list('map_dbl(X_RestTimeFromLastMatch,mysqrt)*X_MatchRelevance')
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(dependencyModelWithPossibleInteractions, 10, 10)
# RSquared: 0.8453, MSE: 2.854117 # BEST
# Abbiamo inserito il quadrato della temperatura perchè migliora i residui


#====================  BEST SUBSET SELECTION WITH INTERACTIONS   ===============

# add non linearities for best subset selection
possibleRelationships = list(
  'I(X_Temperature^2)',
  'X_MatchRelevance*X_AvgGoalConcededLastMatches',
  'X_RestTimeFromLastMatch*X_MatchRelevance'
)    

bestSubsets = bestSubsetSelection(ds, relationships=possibleRelationships, nMSE=10, folds=5, verbose=T, method="exhaustive")
ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

bestSubset = bestSubsets$model[[which.min(bestSubsets$MSE)]]

lm.inspect(bestSubset, 10, 10)


#=============  BEST SUBSETS FOR SELECTED NUMBER OF PREDICTORS   ===============

# add non linearities for best subset selection
possibleRelationships = list(
  'I(X_Temperature^2)',
  'X_MatchRelevance*X_AvgGoalConcededLastMatches',
  'X_RestTimeFromLastMatch*X_MatchRelevance'
)    

N_PREDICTORS_TO_INSPECT = 5
bestSubsets = bestSubsetsByPredictorsNumber(ds, relationships=possibleRelationships, nMSE=10, folds=5, nPredictors=N_PREDICTORS_TO_INSPECT, nSubsets=10, verbose=T)
ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

bestSubset = bestSubsets$model[[which.min(bestSubsets$MSE)]]

lm.inspect(bestSubset, 10, 10)


#============================  RIDGE E LASSO   =================================

# Add non linearities before scaling
bestInteractions = list(
  'I(X_Temperature^2)',
  'X_MatchRelevance*X_AvgGoalConcededLastMatches'
)    
ds_scaled = ds.scale(addNonLinearities(ds, bestInteractions))

lambda_grid = 10^seq(4, -6, length = 10000)

models = lm.shrinkage(ds_scaled, lambda_grid, nMSE=10, folds=10, showPlot=T)
coef(models$lasso$model, s = models$lasso$bestlambda)
coef(models$ridge$model, s = models$ridge$bestlambda)


# predictedY = predictWithGlmnet(models$lasso, newx=as.matrix(ds_scaled[,-1]))

#============================= ELASTIC NET  ===============================

bestInteractions = list(
  'I(X_Temperature^2)',
  'X_MatchRelevance*X_AvgGoalConcededLastMatches'
)  

ds_scaled = ds.scale(addNonLinearities(ds, bestInteractions))

lambda_grid = 10^seq(4, -6, length = 200)
alpha_grid = seq(0,1,length = 100)

best_mse = mean_cvMSE(bestSubset, 10, 10)

MSEs = lm.elasticNet(ds_scaled, alpha_grid, lambda_grid, nMSE=10, folds=10, best_mse = best_mse, showPlot = T, verbose = T)

lm.plotElasticNet(alpha_grid, MSEs, best_mse)

#======================= LINEAR REGRESSION - ISSUES =======================

best_model = lm.byFormulaChunks(ds, list('I(X_Temperature^2)',
                                         'X_Humidity',
                                         'X_SupportersImpact',
                                         'X_MatchRelevance',
                                         'map_dbl(X_RestTimeFromLastMatch,mysqrt)*X_MatchRelevance'
))

# 1) non-linearities & homoschedasticity ----------------------------------
# analyze residuals
plot(best_model, which=1)
# La linea rossa non è dritta quindi c'è della non linearità che non è stata spiegata

# 2) high leverage points -------------------------------------------------
# # compute and plot hat values

hat.plot(best_model)
hats_indices = c(40)

# 4) collinearity ---------------------------------------------------------

# check any collinearity
collinearity_models = vifs.plot(ds)

# 4) outliers -------------------------------------------------------------

outlier_indices = outlier.plot(best_model)

# x) refit ------------------------------------------------------------

indices_to_be_removed = hats_indices

if(length(indices_to_be_removed) > 0) {
  ds = ds[-indices_to_be_removed,]
}
best_model = lm.refit(best_model, ds)
lm.inspect(best_model, 10, 10)

hat.plot(best_model)

#======================= CONCLUSION =======================

best_formula = "Y_PhysicalEndurance ~ I(X_Temperature^2) + X_Humidity + X_SupportersImpact + 
                X_MatchRelevance + map_dbl(X_RestTimeFromLastMatch, mysqrt) * X_MatchRelevance"

best_summary = '
              [1] "================= SUMMARY ================="
              
              Call:
              lm(formula = formula(model), data = data, x = T, y = T)
              
              Residuals:
                  Min      1Q  Median      3Q     Max 
              -2.9886 -1.0088  0.0878  0.8103  2.7920 
              
              Coefficients:
                                                                        Estimate Std. Error t value Pr(>|t|)    
              (Intercept)                                                 5.2606     4.6686   1.127  0.26820    
              I(X_Temperature^2)                                         -5.3580     0.9693  -5.528 4.28e-06 ***
              X_Humidity                                                 -3.1123     1.1088  -2.807  0.00845 ** 
              X_SupportersImpact                                          8.6424     2.8534   3.029  0.00483 ** 
              X_MatchRelevance                                          -19.9788     8.6991  -2.297  0.02833 *  
              map_dbl(X_RestTimeFromLastMatch, mysqrt)                   -8.1673     5.5290  -1.477  0.14941    
              X_MatchRelevance:map_dbl(X_RestTimeFromLastMatch, mysqrt)  32.7400    11.3445   2.886  0.00693 ** 
              ---
              Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
              
              Residual standard error: 1.53 on 32 degrees of freedom
              Multiple R-squared:  0.8265,	Adjusted R-squared:  0.794 
              F-statistic: 25.41 on 6 and 32 DF,  p-value: 7.221e-11
              
              [1] "==================  MSE  =================="
              [1] 2.82
              '

best_model = lm(best_formula, data=ds, y = T, x = T)
exportCOEF(best_model$coefficients)
