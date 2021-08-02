#==============================   CONFIG     ============================

ABS_PATH = 'C:/Users/marco/Documents/UNISA/SDA/progetto/SDAgruppo2'
DATASET_FILENAME = 'RegressionData_final.csv'
Y_LABEL = 'Y_AvgSpeed'
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
# -3.9545 -0.9974  0.0981  0.8226  3.6239 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)                  -0.06732    0.16181  -0.416    0.678    
# X_Temperature                -4.89776    0.15872 -30.858  < 2e-16 ***
#   X_Humidity                   -1.80942    0.18714  -9.669 1.55e-15 ***
#   X_Altitude                    2.18545    0.16052  13.615  < 2e-16 ***
#   X_ClimaticConditions         -3.00731    0.15063 -19.965  < 2e-16 ***
#   X_RestTimeFromLastMatch       5.71595    0.16885  33.853  < 2e-16 ***
#   X_AvgPlayerValue              4.06055    0.16265  24.964  < 2e-16 ***
#   X_MatchRelevance              2.02726    0.15175  13.359  < 2e-16 ***
#   X_AvgGoalConcededLastMatches -0.13026    0.17265  -0.755    0.453    
# X_SupportersImpact            0.03430    0.15377   0.223    0.824    
# X_OpposingSupportersImpact   -0.25869    0.16393  -1.578    0.118    
# ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Residual standard error: 1.556 on 89 degrees of freedom
# Multiple R-squared:  0.9817,	Adjusted R-squared:  0.9796 
# F-statistic: 476.2 on 10 and 89 DF,  p-value: < 2.2e-16
# 
# [1] "==================  MSE  =================="
# [1] 2.984869

#====================== INSPECT RELATIONSHIPS ============================

showPlotsAgainstOutput(ds, 2:(PREDICTORS_NUMBER+1))



#======================== TEST RELATIONSHIPS =============================

possibleDependencies = list('X_MatchRelevance',
                            'X_AvgPlayerValue',
                            'X_RestTimeFromLastMatch',
                            'X_ClimaticConditions',
                            'X_Altitude',
                            'X_Humidity',
                            'X_Temperature'
                            )
dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(dependencyModel, 5)
# RSquared: 0.9811, MSE: 2.721889

#======================== INSPECT INTERACTIONS =============================

# Collect rsquared for every linear model obtained by adding every possible
# interaction between two distinct predictors to the base model.
# Set base rsquared as default value

baseRSquared = summary( lm.byIndices(ds, -1) )$r.squared
interactionMatrix = inspectInteractionMatrix(ds, default=baseRSquared, showHeatmap = T)



#========================  TEST INTERACTIONS   =============================

possibleDependencies = list('X_MatchRelevance',
                            'X_AvgPlayerValue',
                            'X_RestTimeFromLastMatch',
                            'X_ClimaticConditions',
                            'X_Altitude',
                            'X_Humidity',
                            'X_Temperature',
                            'I(X_AvgPlayerValue^2)',
                            'I(X_RestTimeFromLastMatch^2)',
                            'I(X_Humidity^2)')



possibleInteractions = list('X_OpposingSupportersImpact*X_MatchRelevance',
                            'X_Temperature*X_Altitude',
                            'X_RestTimeFromLastMatch*X_ClimaticConditions',
                            'X_AvgPlayerValue*X_AvgGoalConcededLastMatches',
                            'X_Temperature*X_ClimaticConditions'
                            )
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(dependencyModelWithPossibleInteractions, 5)
# RSquared: 0.9874, MSE: 2.368148

#====================  BEST SUBSET SELECTION WITH INTERACTIONS   ===============

# add non linearities for best subset selection
possibleRelationships = list(
  'I(X_AvgPlayerValue^2)',
  'I(X_RestTimeFromLastMatch^2)',
  'I(X_Humidity^2)',
  'X_OpposingSupportersImpact*X_MatchRelevance',
  'X_Temperature*X_Altitude',
  'X_RestTimeFromLastMatch*X_ClimaticConditions',
  'X_AvgPlayerValue*X_AvgGoalConcededLastMatches',
  'X_Temperature*X_ClimaticConditions'
)    

bestSubsets = bestSubsetSelection(ds, relationships=possibleRelationships, nMSE=5, folds=5, verbose=T, method="exhaustive")
ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

bestSubset = bestSubsets$model[[9]]

lm.inspect(bestSubset, 5, 5)

# [1] "================= SUMMARY ================="
# 
# Call:
#   lm(formula = f, data = data, x = TRUE, y = TRUE)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -3.8016 -1.0707  0.0881  1.0320  3.0493 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)                0.5127     0.2194   2.337  0.02165 *  
#   X_Humidity                -1.8322     0.1673 -10.951  < 2e-16 ***
#   X_ClimaticConditions      -3.0026     0.1394 -21.532  < 2e-16 ***
#   X_RestTimeFromLastMatch    5.7283     0.1530  37.444  < 2e-16 ***
#   X_AvgPlayerValue           4.0961     0.1470  27.869  < 2e-16 ***
#   X_MatchRelevance           1.9706     0.1380  14.275  < 2e-16 ***
#   I(X_AvgPlayerValue^2)     -0.5144     0.1581  -3.255  0.00160 ** 
#   X_Temperature             -4.9057     0.1430 -34.297  < 2e-16 ***
#   X_Altitude                 2.1307     0.1457  14.621  < 2e-16 ***
#   X_Temperature:X_Altitude   0.4333     0.1360   3.186  0.00198 ** 
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Residual standard error: 1.426 on 90 degrees of freedom
# Multiple R-squared:  0.9844,	Adjusted R-squared:  0.9829 
# F-statistic: 632.3 on 9 and 90 DF,  p-value: < 2.2e-16
# 
# [1] "==================  MSE  =================="
# [1] 2.424912


#=============  BEST SUBSETS FOR SELECTED NUMBER OF PREDICTORS   ===============

# add non linearities for best subset selection
possibleRelationships = list(
  'I(X_AvgPlayerValue^2)',
  'I(X_RestTimeFromLastMatch^2)',
  'I(X_Humidity^2)',
  'X_OpposingSupportersImpact*X_MatchRelevance',
  'X_Temperature*X_Altitude',
  'X_RestTimeFromLastMatch*X_ClimaticConditions',
  'X_AvgPlayerValue*X_AvgGoalConcededLastMatches',
  'X_Temperature*X_ClimaticConditions'
)    

N_PREDICTORS_TO_INSPECT = 5
bestSubsets = bestSubsetsByPredictorsNumber(ds, relationships=possibleRelationships, nMSE=10, folds=5, nPredictors=N_PREDICTORS_TO_INSPECT, nSubsets=10, verbose=T)
ds.prettyPlot(bestSubsets$MSE, xdata=unlist(map(bestSubsets$model, function(model) summary(model)$r.squared)), xlab="Rank", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

bestSubset = bestSubsets$model[[which.min(bestSubsets$MSE)]]

lm.inspect(bestSubset, 10, 10)


#=================== FORWARD SELECTION WITH INTERACTIONS =======================


possibleRelationships = list(
  'I(X_AvgPlayerValue^2)',
  'I(X_RestTimeFromLastMatch^2)',
  'I(X_Humidity^2)'
)
bestSubsets = bestSubsetSelection(ds, relationships=possibleRelationships, nMSE=10, folds=5, method="forward", nvmax=8, verbose=T)
bestSubset = bestSubsets$model[[which.min(bestSubsets$MSE)]]

ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")


#============================  RIDGE E LASSO   =================================

# Add non linearities before scaling
bestInteractions = list(
  'I(X_AvgPlayerValue^2)',
  'I(X_RestTimeFromLastMatch^2)',
  'I(X_Humidity^2)',
  'X_OpposingSupportersImpact*X_MatchRelevance',
  'X_Temperature*X_Altitude',
  'X_RestTimeFromLastMatch*X_ClimaticConditions',
  'X_AvgPlayerValue*X_AvgGoalConcededLastMatches',
  'X_Temperature*X_ClimaticConditions'
)    
ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)
ds_scaled = addNonLinearities(ds, bestInteractions)

lambda_grid = 10^seq(4, -6, length = 10000)

models = lm.shrinkage(ds_scaled, lambda_grid, nMSE=10, folds=10, showPlot=T)
min(models$ridge$cvm)
coef(models$lasso$model, s = models$lasso$bestlambda)
coef(models$ridge$model, s = models$ridge$bestlambda)


# predictedY = predictWithGlmnet(models$lasso, newx=as.matrix(ds_scaled[,-1]))

#============================= ELASTIC NET  ===============================

bestInteractions = list(
  'I(X_AvgPlayerValue^2)',
  'I(X_RestTimeFromLastMatch^2)',
  'I(X_Humidity^2)',
  'X_OpposingSupportersImpact*X_MatchRelevance',
  'X_Temperature*X_Altitude',
  'X_RestTimeFromLastMatch*X_ClimaticConditions',
  'X_AvgPlayerValue*X_AvgGoalConcededLastMatches',
  'X_Temperature*X_ClimaticConditions'
)    
ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)
ds_scaled = addNonLinearities(ds, bestInteractions)

lambda_grid = 10^seq(4, -6, length = 2000)
alpha_grid = seq(0,1,length = 100)

best_mse = mean_cvMSE(bestSubset, 10, 10)

MSEs = lm.elasticNet(ds_scaled, alpha_grid, lambda_grid, nMSE=5, folds=5, best_mse = best_mse, showPlot = T, verbose = T)

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

# 4) collinearity ---------------------------------------------------------

# check any collinearity
collinearity_models = vifs.plot(ds)

# 4) outliers -------------------------------------------------------------

outlier_indices = outlier.plot(best_model)

# x) refit ------------------------------------------------------------

# indices_to_be_removed = c()
# 
# if(length(indices_to_be_removed) > 0) {
#   ds = ds[-indices_to_be_removed,]
# }
# best_model = lm.refit(best_model, ds)
# 
lm.inspect(best_model, 10, 10)


#======================= CONCLUSION =======================

best_formula = "Y_AvgSpeed ~ X_Humidity + X_ClimaticConditions + X_RestTimeFromLastMatch + 
    X_AvgPlayerValue + X_MatchRelevance + I(X_AvgPlayerValue^2) + 
    X_Temperature * X_Altitude"
best_summary = '
[1] "================= SUMMARY ================="

Call:
lm(formula = f, data = data, x = TRUE, y = TRUE)

Residuals:
    Min      1Q  Median      3Q     Max 
-3.8016 -1.0707  0.0881  1.0320  3.0493 

Coefficients:
                         Estimate Std. Error t value Pr(>|t|)    
(Intercept)                0.5127     0.2194   2.337  0.02165 *  
X_Humidity                -1.8322     0.1673 -10.951  < 2e-16 ***
X_ClimaticConditions      -3.0026     0.1394 -21.532  < 2e-16 ***
X_RestTimeFromLastMatch    5.7283     0.1530  37.444  < 2e-16 ***
X_AvgPlayerValue           4.0961     0.1470  27.869  < 2e-16 ***
X_MatchRelevance           1.9706     0.1380  14.275  < 2e-16 ***
I(X_AvgPlayerValue^2)     -0.5144     0.1581  -3.255  0.00160 ** 
X_Temperature             -4.9057     0.1430 -34.297  < 2e-16 ***
X_Altitude                 2.1307     0.1457  14.621  < 2e-16 ***
X_Temperature:X_Altitude   0.4333     0.1360   3.186  0.00198 ** 
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 1.426 on 90 degrees of freedom
Multiple R-squared:  0.9844,	Adjusted R-squared:  0.9829 
F-statistic: 632.3 on 9 and 90 DF,  p-value: < 2.2e-16

[1] "==================  MSE  =================="
[1] 2.331647
'
best_model = lm(best_formula, data=ds)
exportCOEF(best_model$coefficients)
