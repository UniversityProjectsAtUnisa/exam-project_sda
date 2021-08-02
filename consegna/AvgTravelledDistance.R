#==============================   CONFIG     ============================

ABS_PATH = 'C:/Users/gorra/Desktop/gitSDA_Def/SDAgruppo2'
DATASET_FILENAME = 'RegressionData_final.csv'
Y_LABEL = 'Y_AvgTravelledDistance'
PREDICTORS_NUMBER = 10

#================================ START =================================

setwd(ABS_PATH)
source('./utils.R')
ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)

#==================== REGRESSION WITHOUT INTERACTIONS ====================


baseModel=lm.byIndices(ds, -1)
lm.inspect(baseModel, 10)
#R-squared:  0.9778 -- MSE: 3.134375

#====================== INSPECT RELATIONSHIPS ============================

showPlotsAgainstOutput(ds, 2:(PREDICTORS_NUMBER+1))

#======================== TEST RELATIONSHIPS =============================

possibleDependencies = list('X_Humidity',
                            'X_Altitude',
                            'X_ClimaticConditions',
                            'X_RestTimeFromLastMatch',
                            'X_MatchRelevance',
                            'I(X_Temperature^2)',    
                            'I(X_AvgPlayerValue^3)'
)

dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(dependencyModel, 10)     

# Modello migliore per adesso
# [1] "================= SUMMARY ================="
# 
# Call:
# lm(formula = paste(utils.Y_LABEL, " ~ ", paste(chunks, collapse = " + ")), 
#    data = ds, x = T, y = T)
# 
# Residuals:
# Min      1Q  Median      3Q     Max 
# -4.1096 -1.0705  0.0725  0.9365  4.6632 
# 
# Coefficients:
# Estimate Std. Error t value Pr(>|t|)    
# (Intercept)              0.01588    0.16396   0.097    0.923    
# X_Humidity              -4.74703    0.18775 -25.283  < 2e-16 ***
# X_Altitude              -1.87328    0.16218 -11.551  < 2e-16 ***
# X_ClimaticConditions    -5.07178    0.15443 -32.842  < 2e-16 ***
# X_RestTimeFromLastMatch  3.12186    0.17114  18.241  < 2e-16 ***
# X_MatchRelevance         0.81250    0.15476   5.250 9.72e-07 ***
# X_Temperature           -2.94022    0.15998 -18.379  < 2e-16 ***
# X_AvgPlayerValue         2.96552    0.16393  18.090  < 2e-16 ***
#   ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Residual standard error: 1.6 on 92 degrees of freedom
# Multiple R-squared:  0.9776,	Adjusted R-squared:  0.9758 
# F-statistic: 572.3 on 7 and 92 DF,  p-value: < 2.2e-16
# 
# [1] "==================  MSE  =================="
# [1] 2.930811


#======================== INSPECT INTERACTIONS =============================

# Collect rsquared for every linear model obtained by adding every possible
# interaction between two distinct predictors to the base model.
# Set base rsquared as default value
#
# The matrix will be store as an upper triangular matrix 
# for computational efficiency

baseRSquared = summary( lm.byIndices(ds, -1) )$r.squared
interactionMatrix = inspectInteractionMatrix(ds, default=baseRSquared, showHeatmap = T)


#========================  TEST INTERACTIONS   =============================

possibleDependencies = list('X_Temperature',    
                            'X_AvgPlayerValue', 
                            'X_RestTimeFromLastMatch',
                            'X_Humidity')

possibleInteractions = list('X_RestTimeFromLastMatch*X_MatchRelevance')
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(dependencyModelWithPossibleInteractions, 5)


possibleInteractions = list('X_RestTimeFromLastMatch*X_ClimaticConditions')
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(dependencyModelWithPossibleInteractions, 5)


possibleInteractions = list(
  'X_RestTimeFromLastMatch*X_MatchRelevance', 
  'X_RestTimeFromLastMatch*X_ClimaticConditions'
)
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(dependencyModelWithPossibleInteractions, 5)


#====================  BEST SUBSET SELECTION WITH INTERACTIONS   ===============

# add non linearities for best subset selection
possibleRelationships = list(
  'X_AvgGoalConcededLastMatches*X_Altitude',
  'X_AvgGoalConcededLastMatches*X_Temperature',
  'X_Humidity*X_AvgPlayerValue',
  'I(X_OpposingSupportersImpact^2)'
)    

bestSubsets = bestSubsetSelection(ds, relationships=possibleRelationships, nMSE=10, folds=10, verbose=T, method="exhaustive")
ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

bestSubset = bestSubsets$model[[7]]
lm.inspect(bestSubset,10)

# [1] "================= SUMMARY ================="
# 
# Call:
#   lm(formula = f, data = data, x = TRUE, y = TRUE)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -4.1096 -1.0705  0.0725  0.9365  4.6632 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)              0.01588    0.16396   0.097    0.923    
# X_Temperature           -2.94022    0.15998 -18.379  < 2e-16 ***
#   X_Humidity              -4.74703    0.18775 -25.283  < 2e-16 ***
#   X_Altitude              -1.87328    0.16218 -11.551  < 2e-16 ***
#   X_ClimaticConditions    -5.07178    0.15443 -32.842  < 2e-16 ***
#   X_RestTimeFromLastMatch  3.12186    0.17114  18.241  < 2e-16 ***
#   X_AvgPlayerValue         2.96552    0.16393  18.090  < 2e-16 ***
#   X_MatchRelevance         0.81250    0.15476   5.250 9.72e-07 ***
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Residual standard error: 1.6 on 92 degrees of freedom
# Multiple R-squared:  0.9776,	Adjusted R-squared:  0.9758 
# F-statistic: 572.3 on 7 and 92 DF,  p-value: < 2.2e-16
# 
# [1] "==================  MSE  =================="
# [1] 2.846616

bestSubset = bestSubsets$model[[8]]
lm.inspect(bestSubset,10)
# non c'è miglioria

# IL MIGLIORE:  7 regressori senza interazioni


#============================  RIDGE E LASSO   =================================

# Add non linearities before scaling
bestInteractions = list(
  'X_AvgGoalConcededLastMatches*X_Altitude',
  'X_AvgGoalConcededLastMatches*X_Temperature',
  'X_Humidity*X_AvgPlayerValue',
  'I(X_OpposingSupportersImpact^2)'
)    

ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)
ds_scaled = addNonLinearities(ds, bestInteractions)

lambda_grid = 10^seq(4, -6, length = 2000)

models = lm.shrinkage(ds_scaled, lambda_grid, nMSE=2, folds=4, showPlot=T)
coef(models$lasso$model, s = models$lasso$bestlambda)
coef(models$ridge$model, s = models$ridge$bestlambda)

min(models$ridge$cvm)
# MSE = 2.929349

min(models$lasso$cvm)
# MSE = 2.8 con tutti i predittori
# per interpretabilità meglio best subset


# predictedY = predictWithGlmnet(models$lasso, newx=as.matrix(ds_scaled[,-1]))


#============================= ELASTIC NET  ===============================

bestInteractions = list(
  'X_AvgGoalConcededLastMatches*X_Altitude',
  'X_AvgGoalConcededLastMatches*X_Temperature',
  'X_Humidity*X_AvgPlayerValue',
  'I(X_OpposingSupportersImpact^2)'
)     

ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)
ds_scaled = addNonLinearities(ds, bestInteractions)

lambda_grid = 10^seq(4, -6, length = 500)
alpha_grid = seq(0,1,length = 100)



MSEs = lm.elasticNet(ds_scaled, alpha_grid, lambda_grid, nMSE=10, folds=10, best_mse = 2, showPlot = T, verbose = T)

min(MSEs)
# lm.plotElasticNet(alpha_grid, MSEs, 2)

#======================= LINEAR REGRESSION - ISSUES =======================
# the best model to analyze
best_model = lm(Y_AvgTravelledDistance ~ X_Temperature + X_Humidity + 
                  X_Altitude + X_ClimaticConditions+X_RestTimeFromLastMatch
                +X_AvgPlayerValue+X_MatchRelevance, data=ds,y=T,x=T)

lm.inspect(best_model,10)


# 1) non-linearities & homoschedasticity ----------------------------------
# analyze residuals
plot(best_model, which=1)


# 2) high leverage points -------------------------------------------------

hat.plot(best_model)
hats_indices = c(76)

# 4) collinearity ---------------------------------------------------------

# check any collinearity
collinearity_models = vifs.plot(ds)

# 4) outliers -------------------------------------------------------------

outlier_indices = outlier.plot(best_model)



if(length(outlier_indices) > 0) {
  ds_without_outliers = ds[-outlier_indices,]
}
refitted_best_model = lm.refit(best_model, ds_without_outliers)

lm.inspect(refitted_best_model, 10, 10)
# MSE = 2.734577  R-squared:  0.9777

hat.plot(refitted_best_model)

#plotres(models$lasso$model, s=models$lasso$bestlambda)


#======================= CONCLUSION =======================

best_formula = "Y_AvgTravelledDistance ~ X_Temperature + X_Humidity + X_Altitude + X_ClimaticConditions+X_RestTimeFromLastMatch+X_AvgPlayerValue+X_MatchRelevance"

best_summary = '
      [1] "================= SUMMARY ================="
      
      Call:
      lm(formula = best_formula, data = ds_without_outliers, x = T, 
          y = T)
      
      Residuals:
          Min      1Q  Median      3Q     Max 
      -4.1391 -1.0821  0.1088  0.8481  4.6223 
      
      Coefficients:
                              Estimate Std. Error t value Pr(>|t|)    
      (Intercept)             -0.03321    0.16185  -0.205    0.838    
      X_Temperature           -2.89533    0.15774 -18.354  < 2e-16 ***
      X_Humidity              -4.64296    0.18932 -24.524  < 2e-16 ***
      X_Altitude              -1.90724    0.15936 -11.968  < 2e-16 ***
      X_ClimaticConditions    -5.05454    0.15126 -33.416  < 2e-16 ***
      X_RestTimeFromLastMatch  3.20946    0.17182  18.679  < 2e-16 ***
      X_AvgPlayerValue         2.90540    0.16255  17.874  < 2e-16 ***
      X_MatchRelevance         0.75214    0.15372   4.893 4.27e-06 ***
      ---
      Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
      
      Residual standard error: 1.565 on 91 degrees of freedom
      Multiple R-squared:  0.9777,	Adjusted R-squared:  0.976 
      F-statistic: 570.4 on 7 and 91 DF,  p-value: < 2.2e-16
      
      [1] "==================  MSE  =================="
      [1] 2.645056
'
# OTTENUTO CON BEST SUBSET

best_model = lm(best_formula, data=ds_without_outliers,y=T,x=T)
lm.inspect(best_model, 10, 10)
exportCOEF(best_model$coefficients)
