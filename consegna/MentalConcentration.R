#==============================   CONFIG     ============================

ABS_PATH = 'C:/Users/marco/Documents/UNISA/SDA/progetto/SDAgruppo2'
DATASET_FILENAME = 'RegressionData_final.csv'
Y_LABEL = 'Y_MentalConcentration'
PREDICTORS_NUMBER = 10

#================================ START =================================

setwd(ABS_PATH)
source('./utils.R')
ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)


#==================== REGRESSION WITHOUT INTERACTIONS ====================

baseModel=lm.byIndices(ds, -1)
lm.inspect(baseModel, 5)

# [1] "================= SUMMARY ================="
# 
# Call:
#   lm(formula = f, data = data, x = T, y = T)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -3.2492 -1.1122 -0.0628  1.1345  3.4662 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)                   0.01526    0.16036   0.095    0.924    
# X_Temperature                 0.24103    0.15729   1.532    0.129    
# X_Humidity                    0.08627    0.18546   0.465    0.643    
# X_Altitude                   -0.01143    0.15908  -0.072    0.943    
# X_ClimaticConditions          0.06228    0.14927   0.417    0.678    
# X_RestTimeFromLastMatch       2.92208    0.16733  17.463   <2e-16 ***
#   X_AvgPlayerValue              9.01419    0.16119  55.922   <2e-16 ***
#   X_MatchRelevance              5.98036    0.15039  39.766   <2e-16 ***
#   X_AvgGoalConcededLastMatches  4.32008    0.17110  25.249   <2e-16 ***
#   X_SupportersImpact            1.87229    0.15239  12.286   <2e-16 ***
#   X_OpposingSupportersImpact   -2.98374    0.16246 -18.366   <2e-16 ***
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Residual standard error: 1.542 on 89 degrees of freedom
# Multiple R-squared:  0.9896,	Adjusted R-squared:  0.9885 
# F-statistic: 850.1 on 10 and 89 DF,  p-value: < 2.2e-16
# 
# [1] "==================  MSE  =================="
# [1] 2.827302

#====================== INSPECT RELATIONSHIPS ============================

showPlotsAgainstOutput(ds, 2:(PREDICTORS_NUMBER+1))



#======================== TEST RELATIONSHIPS =============================

possibleDependencies = list(
                            "X_RestTimeFromLastMatch",
                            "X_AvgPlayerValue",
                            "X_MatchRelevance",
                            "X_AvgGoalConcededLastMatches",
                            "X_SupportersImpact",
                            "X_OpposingSupportersImpact"
                            )
dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(dependencyModel, 5)
# RSquared: 0.9893, MSE: 2.518492



#======================== INSPECT INTERACTIONS =============================

# Collect rsquared for every linear model obtained by adding every possible
# interaction between two distinct predictors to the base model.
# Set base rsquared as default value

baseRSquared = summary( lm.byIndices(ds, -1) )$r.squared
interactionMatrix = inspectInteractionMatrix(ds, default=baseRSquared, showHeatmap = T)



#========================  TEST INTERACTIONS   =============================

possibleDependencies = list("X_RestTimeFromLastMatch",
                            "X_AvgPlayerValue",
                            "X_MatchRelevance",
                            "X_AvgGoalConcededLastMatches",
                            "X_SupportersImpact",
                            "X_OpposingSupportersImpact",
                            'I(X_RestTimeFromLastMatch^2)') 



possibleInteractions = list(
                          'X_SupportersImpact*X_AvgGoalConcededLastMatches',
                          'X_OpposingSupportersImpact*X_Humidity',
                          'X_OpposingSupportersImpact*X_Temperature',
                          'X_Humidity*X_Temperature',
                          'X_RestTimeFromLastMatch*X_MatchRelevance'
                            )
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(dependencyModelWithPossibleInteractions, 5)
# RSquared: 0.993, MSE: 2.109674

#====================  BEST SUBSET SELECTION WITH INTERACTIONS   ===============

# add non linearities for best subset selection
possibleRelationships = list(
  'I(X_RestTimeFromLastMatch^2)',
  'X_SupportersImpact*X_AvgGoalConcededLastMatches',
  'X_OpposingSupportersImpact*X_Humidity',
  'X_OpposingSupportersImpact*X_Temperature',
  'X_Humidity*X_Temperature',
  'X_RestTimeFromLastMatch*X_MatchRelevance'
)    

bestSubsets = bestSubsetSelection(ds, relationships=possibleRelationships, nMSE=5, folds=5, verbose=T, method="exhaustive")
ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

bestSubset = bestSubsets$model[[which.min(bestSubsets$MSE)]]

lm.inspect(bestSubset, 10, 10)


#=============  BEST SUBSETS FOR SELECTED NUMBER OF PREDICTORS   ===============

# add non linearities for best subset selection
possibleRelationships = list(
  'I(X_MatchRelevance^2)',
  'I(X_OpposingSupportersImpact^2)',
  'X_AvgPlayerValue*X_SupportersImpact',
  'X_Temperature*X_AvgPlayerValue',
  'X_AvgGoalConcededLastMatches*X_AvgPlayerValue', 
  'X_AvgGoalConcededLastMatches*X_MatchRelevance', 
  'X_AvgGoalConcededLastMatches*X_OpposingSupportersImpact'
)    

N_PREDICTORS_TO_INSPECT = 5
bestSubsets = bestSubsetsByPredictorsNumber(ds, relationships=possibleRelationships, nMSE=10, folds=5, nPredictors=N_PREDICTORS_TO_INSPECT, nSubsets=10, verbose=T)
ds.prettyPlot(bestSubsets$MSE, xdata=unlist(map(bestSubsets$model, function(model) summary(model)$r.squared)), xlab="Rank", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

bestSubset = bestSubsets$model[[8]]

lm.inspect(bestSubset, 10, 10)


#=================== FORWARD SELECTION WITH INTERACTIONS =======================


possibleRelationships = list(
  'I(X_MatchRelevance^2)',
  'I(X_OpposingSupportersImpact^2)'
)
bestSubsets = bestSubsetSelection(ds, relationships=possibleRelationships, nMSE=10, folds=5, method="forward", nvmax=8, verbose=T)
bestSubset = bestSubsets$model[[which.min(bestSubsets$MSE)]]

ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")


#============================  RIDGE E LASSO   =================================

# Add non linearities before scaling
bestInteractions = list(
  'I(X_MatchRelevance^2)',
  'I(X_OpposingSupportersImpact^2)',
  'X_AvgPlayerValue*X_SupportersImpact',
  'X_Temperature*X_AvgPlayerValue',
  'X_AvgGoalConcededLastMatches*X_AvgPlayerValue', 
  'X_AvgGoalConcededLastMatches*X_MatchRelevance', 
  'X_AvgGoalConcededLastMatches*X_OpposingSupportersImpact'
)    

ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)
ds_scaled = addNonLinearities(ds, bestInteractions)

lambda_grid = 10^seq(4, -6, length = 10000)

models = lm.shrinkage(ds_scaled, lambda_grid, nMSE=10, folds=10, showPlot=T)
min(models$lasso$cvm)
coef(models$lasso$model, s = models$lasso$bestlambda)
coef(models$ridge$model, s = models$ridge$bestlambda)


# predictedY = predictWithGlmnet(models$lasso, newx=as.matrix(ds_scaled[,-1]))

#============================= ELASTIC NET  ===============================

bestInteractions = list(
  'I(X_MatchRelevance^2)',
  'I(X_OpposingSupportersImpact^2)',
  'X_AvgPlayerValue*X_SupportersImpact',
  'X_Temperature*X_AvgPlayerValue',
  'X_AvgGoalConcededLastMatches*X_AvgPlayerValue', 
  'X_AvgGoalConcededLastMatches*X_MatchRelevance', 
  'X_AvgGoalConcededLastMatches*X_OpposingSupportersImpact'
)    

ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)
ds_scaled = addNonLinearities(ds, bestInteractions)

lambda_grid = 10^seq(4, -6, length = 2000)
alpha_grid = seq(0,1,length = 100)

best_mse = mean_cvMSE(bestSubset, 10, 10)

MSEs = lm.elasticNet(ds_scaled, alpha_grid, lambda_grid, nMSE=10, folds=10, best_mse = best_mse, showPlot = T, verbose = T)

lm.plotElasticNet(alpha_grid, MSEs, best_mse)

#======================= LINEAR REGRESSION - ISSUES =======================

# the best model to analyze
best_model = bestSubset # or any other (not glmnet model!)

# 1) non-linearities & homoschedasticity ----------------------------------
# analyze residuals
plot(best_model, which=1)
# La linea rossa non � dritta quindi c'� della non linearit� che non � stata spiegata

# 2) high leverage points -------------------------------------------------
# # compute and plot hat values

hat.plot(best_model)
hats_indices = c(4)

# 4) collinearity ---------------------------------------------------------

# check any collinearity
collinearity_models = vifs.plot(ds)

# 4) outliers -------------------------------------------------------------

outlier_indices = outlier.plot(best_model)
outlier_indices = c(3)

# x) refit ------------------------------------------------------------

indices_to_be_removed = c(12, 44)

if(length(indices_to_be_removed) > 0) {
  ds = ds[-indices_to_be_removed,]
}
best_model = lm.refit(best_model, ds)

lm.inspect(best_model, 10, 10)


#======================= CONCLUSION =======================

best_formula = "Y_MentalConcentration ~ X_RestTimeFromLastMatch + X_AvgPlayerValue + 
    X_MatchRelevance + X_OpposingSupportersImpact + I(X_RestTimeFromLastMatch^2) + 
    X_SupportersImpact * X_AvgGoalConcededLastMatches"
best_summary = '
[1] "================= SUMMARY ================="

Call:
lm(formula = formula(model), data = data, x = T, y = T)

Residuals:
    Min      1Q  Median      3Q     Max 
-2.6380 -0.9439 -0.1142  0.9234  3.7403 

Coefficients:
                                                Estimate Std. Error t value Pr(>|t|)    
(Intercept)                                       0.5762     0.2144   2.688 0.008584 ** 
X_RestTimeFromLastMatch                           2.8401     0.1546  18.371  < 2e-16 ***
X_AvgPlayerValue                                  9.1769     0.1511  60.741  < 2e-16 ***
X_MatchRelevance                                  6.0275     0.1400  43.046  < 2e-16 ***
X_OpposingSupportersImpact                       -3.0295     0.1535 -19.736  < 2e-16 ***
I(X_RestTimeFromLastMatch^2)                     -0.5958     0.1730  -3.444 0.000877 ***
X_SupportersImpact                                1.7301     0.1504  11.501  < 2e-16 ***
X_AvgGoalConcededLastMatches                      4.1430     0.1553  26.674  < 2e-16 ***
X_SupportersImpact:X_AvgGoalConcededLastMatches  -0.5393     0.1511  -3.570 0.000578 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 1.413 on 89 degrees of freedom
Multiple R-squared:  0.9912,	Adjusted R-squared:  0.9904 
F-statistic:  1254 on 8 and 89 DF,  p-value: < 2.2e-16

[1] "==================  MSE  =================="
[1] 2.224745
'
exportCOEF(best_model$coefficients)
