#==============================   CONFIG     ============================

ABS_PATH = 'C:/Users/marco/Documents/UNISA/SDA/progetto/SDAgruppo2'
DATASET_FILENAME = 'RegressionData_SDA_AH_group2.csv'
Y_LABEL = 'Y_PressingCapability'
PREDICTORS_NUMBER = 10

#================================ START =================================

setwd(ABS_PATH)
source('./utils.R')
ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)

for (a_ in 1:10000){
  ds = rbind(ds, runif(n = ncol(ds), min = 1, max = 10))
}

#==================== REGRESSION WITHOUT INTERACTIONS ====================

baseModel=lm.byIndices(ds, -1)
lm.inspect(baseModel, 5)


#====================== INSPECT RELATIONSHIPS ============================

showPlotsAgainstOutput(ds, 2:(PREDICTORS_NUMBER+1))



#======================== TEST RELATIONSHIPS =============================

possibleDependencies = list(
                            'X_Temperature', 
                            'I(X_Temperature^2)',       
                            'X_RestTimeFromLastMatch', 
                            'I(X_RestTimeFromLastMatch^2)',    
                            'X_AvgPlayerValue', 
                            'I(X_AvgPlayerValue^2)',     
                            'X_MatchRelevance',
                            'I(X_MatchRelevance^2)',    
                            'X_SupportersImpact',
                            'I(X_SupportersImpact^2)',  
                            'X_OpposingSupportersImpact',
                            'I(X_OpposingSupportersImpact^2)'
                            )
dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(dependencyModel, 5)
# RSquared: 0.8089, MSE: 4.230247


possibleDependencies = list('X_Temperature',       
                            'X_RestTimeFromLastMatch', 
                            'X_AvgPlayerValue',    
                            'X_MatchRelevance',
                            'X_SupportersImpact'
                            )
dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(dependencyModel, 5)
# RSquared: 0.8249, MSE: 2.381227

#======================== INSPECT INTERACTIONS =============================

# Collect rsquared for every linear model obtained by adding every possible
# interaction between two distinct predictors to the base model.
# Set base rsquared as default value

baseRSquared = summary( lm.byIndices(ds, -1) )$r.squared
interactionMatrix = inspectInteractionMatrix(ds, default=baseRSquared, showHeatmap = T)



#========================  TEST INTERACTIONS   =============================

possibleDependencies = list('X_Temperature',       
                            'X_RestTimeFromLastMatch', 
                            'X_AvgPlayerValue',    
                            'X_MatchRelevance',
                            'X_SupportersImpact',
                            'I(X_RestTimeFromLastMatch^2)',
                            'I(X_OpposingSupportersImpact^2)'
                            )



possibleInteractions = list('X_Altitude*X_SupportersImpact')
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(dependencyModelWithPossibleInteractions, 5)


possibleInteractions = list('X_Altitude*X_SupportersImpact', 
                            'X_Altitude*X_Humidity'
                            )
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(dependencyModelWithPossibleInteractions, 5)


possibleInteractions = list('X_Altitude*X_Humidity')
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(dependencyModelWithPossibleInteractions, 5)

#====================  BEST SUBSET SELECTION WITH INTERACTIONS   ===============

# add non linearities for best subset selection
possibleRelationships = list(
  'I(X_RestTimeFromLastMatch^2)',
  'I(X_OpposingSupportersImpact^2)',
  'X_ClimaticConditions*X_RestTimeFromLastMatch',
  'X_Altitude*X_SupportersImpact', 
  'X_Altitude*X_AvgPlayerValue', 
  'X_Altitude*X_Humidity'
)    

start.time <- Sys.time()
bestSubsets = bestSubsetSelection(ds, relationships=possibleRelationships, nMSE=10, folds=5, verbose=T, method="exhaustive")
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken
ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

bestSubset = bestSubsets$model[[which.min(bestSubsets$MSE)]]

lm.inspect(bestSubset, 10, 10)

# [1] "================= SUMMARY ================="
# 
# Call:
#   lm(formula = f, data = data, x = TRUE, y = TRUE)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -3.2384 -0.8265  0.1915  0.9496  2.4500 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)                     7.7785     4.0278   1.931 0.062358 .  
# X_Temperature                  -3.5865     0.9873  -3.633 0.000971 ***
#   X_Altitude                    -19.8982     6.2588  -3.179 0.003269 ** 
#   X_RestTimeFromLastMatch         4.2249     0.7471   5.655 2.95e-06 ***
#   X_AvgPlayerValue                5.9285     0.9934   5.968 1.19e-06 ***
#   X_MatchRelevance                6.6044     2.5975   2.543 0.016039 *  
#   X_SupportersImpact             -9.3866     6.7205  -1.397 0.172114    
# X_Altitude:X_SupportersImpact  37.0298    12.1193   3.055 0.004507 ** 
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Residual standard error: 1.452 on 32 degrees of freedom
# Multiple R-squared:  0.8018,	Adjusted R-squared:  0.7585 
# F-statistic:  18.5 on 7 and 32 DF,  p-value: 1.348e-09
# 
# [1] "==================  MSE  =================="
# [1] 2.874053



#=============  BEST SUBSETS FOR SELECTED NUMBER OF PREDICTORS   ===============

# add non linearities for best subset selection
possibleRelationships = list(
  'I(X_RestTimeFromLastMatch^2)',
  'I(X_OpposingSupportersImpact^2)',
  'X_ClimaticConditions*X_RestTimeFromLastMatch',
  'X_Altitude*X_SupportersImpact', 
  'X_Altitude*X_AvgPlayerValue', 
  'X_Altitude*X_Humidity'
)    

N_PREDICTORS_TO_INSPECT = 7
bestSubsets = bestSubsetsByPredictorsNumber(ds, relationships=possibleRelationships, nMSE=10, folds=5, nPredictors=N_PREDICTORS_TO_INSPECT, nSubsets=10, verbose=T)
ds.prettyPlot(bestSubsets$MSE, xlab="Rank", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

bestSubset = bestSubsets$model[[which.min(bestSubsets$MSE)]]

lm.inspect(bestSubset, 10, 10)

# [1] "================= SUMMARY ================="
# 
# Call:
#   lm(formula = f, data = data, x = TRUE, y = TRUE)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -3.2384 -0.8265  0.1915  0.9496  2.4500 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)                     7.7785     4.0278   1.931 0.062358 .  
# X_Temperature                  -3.5865     0.9873  -3.633 0.000971 ***
#   X_RestTimeFromLastMatch         4.2249     0.7471   5.655 2.95e-06 ***
#   X_AvgPlayerValue                5.9285     0.9934   5.968 1.19e-06 ***
#   X_MatchRelevance                6.6044     2.5975   2.543 0.016039 *  
#   X_Altitude                    -19.8982     6.2588  -3.179 0.003269 ** 
#   X_SupportersImpact             -9.3866     6.7205  -1.397 0.172114    
# X_Altitude:X_SupportersImpact  37.0298    12.1193   3.055 0.004507 ** 
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Residual standard error: 1.452 on 32 degrees of freedom
# Multiple R-squared:  0.8018,	Adjusted R-squared:  0.7585 
# F-statistic:  18.5 on 7 and 32 DF,  p-value: 1.348e-09
# 
# [1] "==================  MSE  =================="
# [1] 2.839954

#=================== FORWARD SELECTION WITH INTERACTIONS =======================


possibleRelationships = list(
  "I(X_Temperature^2)",
  "I(X_Humidity^2)",
  "I(X_Altitude^2)",
  "I(X_ClimaticConditions^2)",
  "I(X_RestTimeFromLastMatch^2)", 
  "I(X_AvgPlayerValue^2)",
  "I(X_MatchRelevance^2)",
  "I(X_AvgGoalConcededLastMatches^2)",
  "I(X_SupportersImpact^2)",
  "I(X_OpposingSupportersImpact^2)"  
)
bestSubsets = bestSubsetSelection(ds, relationships=possibleRelationships, nMSE=10, folds=5, method="forward", nvmax=8, verbose=T)
bestSubset = bestSubsets$model[[which.min(bestSubsets$MSE)]]

ds.prettyPlot(bestSubsets$MSE, xdata=unlist(map(bestSubsets$model, function(model) summary(model)$r.squared)), xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")


#============================  RIDGE E LASSO   =================================

# Add non linearities before scaling
bestInteractions = list(
  'I(X_RestTimeFromLastMatch^2)',
  'I(X_OpposingSupportersImpact^2)',
  'X_ClimaticConditions*X_RestTimeFromLastMatch',
  'X_Altitude*X_SupportersImpact', 
  'X_Altitude*X_AvgPlayerValue', 
  'X_Altitude*X_Humidity'
)   

ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER) 
ds_scaled = addNonLinearities(ds, bestInteractions)

lambda_grid = 10^seq(4, -6, length = 10000)

models = lm.shrinkage(ds_scaled, lambda_grid, nMSE=10, folds=10, showPlot=T)
min(models$ridge$cvm)
coef(models$lasso$model, s = models$lasso$bestlambda)
coef(models$ridge$model, s = models$ridge$bestlambda)

# BEST IN ASSOLUTO: PER LA PRIMA VOLTA RIDGE VA MEGLIO !!!!!!!!
# s1
# (Intercept)                                  10.72554847
# X_Temperature                                -0.27553818
# X_Humidity                                   -0.02425850
# X_Altitude                                   -0.32459382
# X_ClimaticConditions                         -0.08712657
# X_RestTimeFromLastMatch                       0.47367753
# X_AvgPlayerValue                              1.43017436
# X_MatchRelevance                              0.96249278
# X_AvgGoalConcededLastMatches                  0.28148484
# X_SupportersImpact                            0.38279557
# X_OpposingSupportersImpact                   -0.39000564
# I(X_RestTimeFromLastMatch^2)                  0.51035815
# I(X_OpposingSupportersImpact^2)              -0.01914322
# X_ClimaticConditions*X_RestTimeFromLastMatch  0.24163688
# X_Altitude*X_SupportersImpact                -0.22311179
# X_Altitude*X_AvgPlayerValue                   1.08411235
# X_Altitude*X_Humidity                        -0.18467249


#============================= ELASTIC NET  ===============================

bestInteractions = list(
  'I(X_RestTimeFromLastMatch^2)',
  'I(X_OpposingSupportersImpact^2)',
  'X_ClimaticConditions*X_RestTimeFromLastMatch',
  'X_Altitude*X_SupportersImpact', 
  'X_Altitude*X_AvgPlayerValue', 
  'X_Altitude*X_Humidity'
)    

ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)
ds_scaled = addNonLinearities(ds, bestInteractions)

lambda_grid = 10^seq(4, -6, length = 2000)
alpha_grid = seq(0,1,length = 100)

best_mse = mean_cvMSE(bestSubset, 10, 10)

MSEs = lm.elasticNet(ds_scaled, alpha_grid, lambda_grid, nMSE=10, folds=10, best_mse = best_mse, showPlot = T, verbose = T)

lm.plotElasticNet(alpha_grid, MSEs, best_mse)

#======================= LINEAR REGRESSION - ISSUES =======================

best_model = lm.byFormulaChunks(ds, list(
  "X_Temperature",
  "X_RestTimeFromLastMatch",
  "X_AvgPlayerValue",
  "X_MatchRelevance",
  "X_Altitude",
  "X_SupportersImpact",
  "X_Altitude*X_SupportersImpact"
))

# the best model to analyze
best_model = bestSubset # or any other (not glmnet model!)

# 1) non-linearities & homoschedasticity ----------------------------------
# analyze residuals
plot(best_model, which=1)
# La linea rossa non � dritta quindi c'� della non linearit� che non � stata spiegata

# 2) high leverage points -------------------------------------------------
# # compute and plot hat values

hat.plot(best_model)
hats_indices = c(40)

# 4) collinearity ---------------------------------------------------------

# check any collinearity
collinearity_models = vifs.plot(ds)

# 4) outliers -------------------------------------------------------------

outlier_indices = outlier.plot(best_model)
outlier_indices = c(3)

# x) refit ------------------------------------------------------------

indices_to_be_removed = c(32)

if(length(indices_to_be_removed) > 0) {
  ds = ds[-indices_to_be_removed,]
}
best_model = lm.refit(best_model, ds)

lm.inspect(best_model, 10, 10)


#======================= CONCLUSION =======================

best_formula = "Y_PressingCapability ~ X_Temperature + X_RestTimeFromLastMatch + 
    X_AvgPlayerValue + X_MatchRelevance + X_Altitude + X_SupportersImpact + 
    X_Altitude * X_SupportersImpact"
best_summary = '
[1] "================= SUMMARY ================="

Call:
lm(formula = formula(model), data = data, x = T, y = T)

Residuals:
    Min      1Q  Median      3Q     Max 
-3.0979 -0.8151  0.2151  0.9885  2.3031 

Coefficients:
                              Estimate Std. Error t value Pr(>|t|)    
(Intercept)                     7.7389     4.0440   1.914  0.06493 .  
X_Temperature                  -3.3043     1.0435  -3.167  0.00345 ** 
X_RestTimeFromLastMatch         4.2696     0.7518   5.679 3.07e-06 ***
X_AvgPlayerValue                5.7459     1.0194   5.636 3.46e-06 ***
X_MatchRelevance                6.6053     2.6078   2.533  0.01659 *  
X_Altitude                    -19.3112     6.3201  -3.056  0.00459 ** 
X_SupportersImpact             -9.4490     6.7475  -1.400  0.17133    
X_Altitude:X_SupportersImpact  35.7317    12.2594   2.915  0.00656 ** 
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 1.457 on 31 degrees of freedom
Multiple R-squared:  0.7823,	Adjusted R-squared:  0.7332 
F-statistic: 15.92 on 7 and 31 DF,  p-value: 1.142e-08

[1] "==================  MSE  =================="
[1] 3.025554
'
best_model = lm(best_formula, data=ds)
exportCOEF(coef(models$ridge$model, s = models$ridge$bestlambda), T)

# ds_pre_scale = addNonLinearities(ds, bestInteractions)

# retval = scale(ds_pre_scale)
# ds_scaled
# finalds = read.csv("FINAL.csv")
# finalds = addNonLinearities(finalds, bestInteractions)

# ds_pre_scale = (ds_pre_scale-medie)/variances
# var(ds_pre_scale['X_Temperature'])

# finalds = (finalds-medie)/variances

# ds_pre_scale = addNonLinearities(ds, bestInteractions)

# medie = colMeans(ds_pre_scale)
# sds = colSds(as.matrix(ds_pre_scale))

# ds.scale = function(ds) {
#   # Assumes the only dependent variable is 
#   # in the first column
#   return(scale(ds[1:length(ds)]))
# }

# predict(models$ridge$model, s=models$ridge$bestlambda, newx=as.matrix(as.data.frame(new)), type="response")

# new = (finalds[1] - medie[2])/sds[2]
# for (i in 2:ncol(finalds)) {
#     new = cbind(new, (finalds[i] - medie[i+1])/sds[i+1])
# }
