#==============================   CONFIG     ============================

ABS_PATH = 'C:/Users/marco/Documents/UNISA/SDA/progetto/SDAgruppo2'
DATASET_FILENAME = 'RegressionData_final.csv'
Y_LABEL = 'Y_PressingCapability'
PREDICTORS_NUMBER = 10

#================================ START =================================

setwd(ABS_PATH)
source('./utils.R')
ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)

#==================== REGRESSION WITHOUT INTERACTIONS ====================

baseModel=lm.byIndices(ds, -1)
lm.inspect(baseModel, 10, 10)

'[1] "================= SUMMARY ================="

Call:
lm(formula = f, data = data, x = T, y = T)

Residuals:
    Min      1Q  Median      3Q     Max 
-3.6467 -1.0333 -0.0055  1.2385  4.4370 

Coefficients:
                             Estimate Std. Error t value Pr(>|t|)    
(Intercept)                  -0.03174    0.17001  -0.187    0.852    
X_Temperature                -0.98848    0.16675  -5.928 5.72e-08 ***
X_Humidity                   -0.02227    0.19662  -0.113    0.910    
X_Altitude                   -0.17045    0.16865  -1.011    0.315    
X_ClimaticConditions         -1.06987    0.15826  -6.760 1.39e-09 ***
X_RestTimeFromLastMatch       4.82436    0.17740  27.195  < 2e-16 ***
X_AvgPlayerValue              6.13091    0.17089  35.876  < 2e-16 ***
X_MatchRelevance              8.03031    0.15943  50.367  < 2e-16 ***
X_AvgGoalConcededLastMatches  0.92545    0.18139   5.102 1.88e-06 ***
X_SupportersImpact            2.05277    0.16156  12.706  < 2e-16 ***
X_OpposingSupportersImpact   -0.83955    0.17224  -4.874 4.73e-06 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 1.635 on 89 degrees of freedom
Multiple R-squared:  0.9867,	Adjusted R-squared:  0.9852 
F-statistic:   662 on 10 and 89 DF,  p-value: < 2.2e-16

[1] "==================  MSE  =================="
[1] 3.002713'


#====================== INSPECT RELATIONSHIPS ============================

showPlotsAgainstOutput(ds, 2:(PREDICTORS_NUMBER+1))



#======================== TEST RELATIONSHIPS =============================

possibleDependencies = list(
                            'X_Temperature',    
                            'X_ClimaticConditions',
                            'X_RestTimeFromLastMatch', 
                            'I(X_RestTimeFromLastMatch^2)',   
                            'X_AvgPlayerValue',   
                            'X_MatchRelevance', 
                            'X_AvgGoalConcededLastMatches',
                            'X_SupportersImpact',
                            'X_OpposingSupportersImpact',
                            'I(X_OpposingSupportersImpact^2)'
                            )
dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(dependencyModel, 5)


possibleDependencies = list(
                            'X_Temperature',    
                            'X_ClimaticConditions',
                            'X_RestTimeFromLastMatch',    
                            'X_AvgPlayerValue',   
                            'X_MatchRelevance', 
                            'X_AvgGoalConcededLastMatches',
                            'X_SupportersImpact',
                            'X_OpposingSupportersImpact',
                            'I(X_OpposingSupportersImpact^2)'
                            )
dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(dependencyModel, 5)
# RSquared: 0.9872, MSE: 2.881993


possibleDependencies = list(
  'X_Temperature',    
  'X_ClimaticConditions',
  'X_RestTimeFromLastMatch', 
  'I(X_AvgPlayerValue^2)',   
  'X_AvgPlayerValue',   
  'X_MatchRelevance', 
  'X_AvgGoalConcededLastMatches',
  'X_SupportersImpact',
  'X_OpposingSupportersImpact',
  'I(X_OpposingSupportersImpact^2)'
)
dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(dependencyModel, 5)
# RSquared: 0.9879, MSE: 2.838846

#======================== INSPECT INTERACTIONS =============================

# Collect rsquared for every linear model obtained by adding every possible
# interaction between two distinct predictors to the base model.
# Set base rsquared as default value

baseRSquared = summary( lm.byIndices(ds, -1) )$r.squared
interactionMatrix = inspectInteractionMatrix(ds, default=baseRSquared, showHeatmap = T)



#========================  TEST INTERACTIONS   =============================

possibleDependencies = list(
                            'X_Temperature',    
                            'X_ClimaticConditions',
                            'X_RestTimeFromLastMatch', 
                            'I(X_AvgPlayerValue^2)',   
                            'X_AvgPlayerValue',   
                            'X_MatchRelevance', 
                            'X_AvgGoalConcededLastMatches',
                            'X_SupportersImpact',
                            'X_OpposingSupportersImpact',
                            'I(X_OpposingSupportersImpact^2)'
                            )



possibleInteractions = list('X_RestTimeFromLastMatch*X_OpposingSupportersImpact')
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(dependencyModelWithPossibleInteractions, 5)


possibleInteractions = list('X_RestTimeFromLastMatch*X_OpposingSupportersImpact',
                            'X_AvgGoalConcededLastMatches*X_Humidity'
                            )
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(dependencyModelWithPossibleInteractions, 5)


possibleInteractions = list('X_RestTimeFromLastMatch*X_OpposingSupportersImpact',
                            'X_AvgGoalConcededLastMatches*X_Humidity',
                            'X_Altitude*X_RestTimeFromLastMatch'
                            )
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(dependencyModelWithPossibleInteractions, 5)

#====================  BEST SUBSET SELECTION WITH INTERACTIONS   ===============

# add non linearities for best subset selection
possibleRelationships = list(
  'I(X_AvgPlayerValue^2)',
  'I(X_OpposingSupportersImpact^2)',
  'I(X_Temperature^2)',
  'X_RestTimeFromLastMatch*X_OpposingSupportersImpact',
  'X_AvgGoalConcededLastMatches*X_Humidity',
  'X_Altitude*X_RestTimeFromLastMatch'
)    

bestSubsets = bestSubsetSelection(ds, relationships=possibleRelationships, nMSE=10, folds=5, verbose=T, method="exhaustive")
ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

bestSubset = bestSubsets$model[[10]]

lm.inspect(bestSubset, 10, 10)

# [1] "================= SUMMARY ================="
# 
# Call:
#   lm(formula = f, data = data, x = TRUE, y = TRUE)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -3.1699 -0.9574 -0.0742  0.8877  4.2548 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)                                         -0.4759     0.2377  -2.002  0.04836 *  
#   X_Temperature                                       -0.9226     0.1554  -5.937 5.50e-08 ***
#   X_ClimaticConditions                                -1.0942     0.1445  -7.575 3.21e-11 ***
#   X_AvgPlayerValue                                     6.1266     0.1607  38.127  < 2e-16 ***
#   X_MatchRelevance                                     7.9999     0.1506  53.124  < 2e-16 ***
#   X_AvgGoalConcededLastMatches                         0.9607     0.1661   5.786 1.06e-07 ***
#   X_SupportersImpact                                   2.0948     0.1525  13.734  < 2e-16 ***
#   I(X_AvgPlayerValue^2)                                0.3824     0.1702   2.246  0.02718 *  
#   X_RestTimeFromLastMatch                              4.7379     0.1677  28.249  < 2e-16 ***
#   X_OpposingSupportersImpact                          -0.6892     0.1691  -4.076 9.93e-05 ***
#   X_RestTimeFromLastMatch:X_OpposingSupportersImpact   0.5148     0.1852   2.779  0.00665 ** 
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Residual standard error: 1.541 on 89 degrees of freedom
# Multiple R-squared:  0.9882,	Adjusted R-squared:  0.9869 
# F-statistic: 746.1 on 10 and 89 DF,  p-value: < 2.2e-16
# 
# [1] "==================  MSE  =================="
# [1] 2.713025




#=============  BEST SUBSETS FOR SELECTED NUMBER OF PREDICTORS   ===============

# add non linearities for best subset selection
possibleRelationships = list(
  'I(X_AvgPlayerValue^2)',
  'I(X_OpposingSupportersImpact^2)',
  'I(X_Temperature^2)',
  'X_RestTimeFromLastMatch*X_OpposingSupportersImpact',
  'X_AvgGoalConcededLastMatches*X_Humidity',
  'X_Altitude*X_RestTimeFromLastMatch'
)    

N_PREDICTORS_TO_INSPECT = 7
bestSubsets = bestSubsetsByPredictorsNumber(ds, relationships=possibleRelationships, nMSE=10, folds=5, nPredictors=N_PREDICTORS_TO_INSPECT, nSubsets=10, verbose=T)
ds.prettyPlot(bestSubsets$MSE, xlab="Rank", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

bestSubset = bestSubsets$model[[which.min(bestSubsets$MSE)]]

lm.inspect(bestSubset, 10, 10)


#=================== FORWARD SELECTION WITH INTERACTIONS =======================


possibleRelationships = list(
  'I(X_AvgPlayerValue^2)',
  'I(X_OpposingSupportersImpact^2)',
  'I(X_Temperature^2)',
)
bestSubsets = bestSubsetSelection(ds, relationships=possibleRelationships, nMSE=10, folds=5, method="forward", nvmax=8, verbose=T)
bestSubset = bestSubsets$model[[which.min(bestSubsets$MSE)]]

ds.prettyPlot(bestSubsets$MSE, xdata=unlist(map(bestSubsets$model, function(model) summary(model)$r.squared)), xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")


#============================  RIDGE E LASSO   =================================

# Add non linearities before scaling
bestInteractions = list(
  'I(X_AvgPlayerValue^2)',
  'I(X_OpposingSupportersImpact^2)',
  'I(X_Temperature^2)',
  'X_RestTimeFromLastMatch*X_OpposingSupportersImpact',
  'X_AvgGoalConcededLastMatches*X_Humidity',
  'X_Altitude*X_RestTimeFromLastMatch'
)   

ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER) 
ds_scaled = addNonLinearities(ds, bestInteractions)

lambda_grid = 10^seq(4, -6, length = 10000)

models = lm.shrinkage(ds_scaled, lambda_grid, nMSE=10, folds=10, showPlot=T)
min(models$ridge$cvm)
models$ridge$bestlambda
coef(models$lasso$model, s = models$lasso$bestlambda)
coef(models$ridge$model, s = models$ridge$bestlambda)

#============================= ELASTIC NET  ===============================

bestInteractions = list(
  'I(X_AvgPlayerValue^2)',
  'I(X_OpposingSupportersImpact^2)',
  'I(X_Temperature^2)',
  'X_RestTimeFromLastMatch*X_OpposingSupportersImpact',
  'X_AvgGoalConcededLastMatches*X_Humidity',
  'X_Altitude*X_RestTimeFromLastMatch'
)    

ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)
ds_scaled = addNonLinearities(ds, bestInteractions)

lambda_grid = 10^seq(4, -6, length = 2000)
alpha_grid = seq(0,1,length = 100)

best_mse = mean_cvMSE(bestSubset, 10, 10)

MSEs = lm.elasticNet(ds_scaled, alpha_grid, lambda_grid, nMSE=5, folds=5, best_mse = best_mse, showPlot = T, verbose = T)

lm.plotElasticNet(alpha_grid, MSEs, best_mse)

#======================= CONCLUSION =======================

exportCOEF(coef(models$ridge$model, s = models$ridge$bestlambda), T)
