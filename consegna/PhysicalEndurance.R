#==============================   CONFIG     ============================

ABS_PATH = 'C:/Users/gorra/Desktop/gitSDA_Def/SDAgruppo2'
DATASET_FILENAME = 'RegressionData_final.csv'
Y_LABEL = 'Y_PhysicalEndurance'
PREDICTORS_NUMBER = 10

#================================ START =================================

setwd(ABS_PATH)
source('./utils.R')
ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)


#==================== REGRESSION WITHOUT INTERACTIONS ====================

baseModel=lm.byIndices(ds, -1)
lm.inspect(baseModel, 10)
#R-squared:  0.9803  MSE=2.171863

#====================== INSPECT RELATIONSHIPS ============================

showPlotsAgainstOutput(ds, 2:(PREDICTORS_NUMBER+1))



#======================== TEST RELATIONSHIPS =============================

possibleDependencies = list('X_Temperature',
                            'exp(-X_RestTimeFromLastMatch)',
                            'X_Humidity',
                            'X_SupportersImpact',
                            'X_Altitude',
                            'X_AvgPlayerValue',
                            'X_AvgGoalConcededLastMatches',
                            'X_SupportersImpact',
                            'X_OpposingSupportersImpact'
)

dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(dependencyModel, 10)
# RSquared: 0.9158, MSE: 9.11039

#======================== INSPECT INTERACTIONS =============================

# Collect rsquared for every linear model obtained by adding every possible
# interaction between two distinct predictors to the base model.
# Set base rsquared as default value

baseRSquared = summary( lm.byIndices(ds, -1) )$r.squared
interactionMatrix = inspectInteractionMatrix(ds, default=baseRSquared, showHeatmap = T)


#====================  BEST SUBSET SELECTION WITH INTERACTIONS   ===============

# add non linearities for best subset selection
possibleRelationships = list(
  'X_SupportersImpact*X_OpposingSupportersImpact',
  'X_SupportersImpact*X_MatchRelevance',
  'I(X_AvgPlayerValue^2)'
)    

bestSubsets = bestSubsetSelection(ds, relationships=possibleRelationships, nMSE=10, folds=5, verbose=T, method="exhaustive")
ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")


bestSubset = bestSubsets$model[[7]]
bestSubsetMSE = bestSubsets$MSE[[7]]
lm.inspect(bestSubset, 10, 10)

# [1] "================= SUMMARY ================="
# 
# Call:
# lm(formula = f, data = data, x = TRUE, y = TRUE)
# 
# Residuals:
# Min      1Q  Median      3Q     Max 
# -4.4883 -0.8373 -0.0209  0.9551  4.1173 
# 
# Coefficients:
# Estimate Std. Error t value Pr(>|t|)    
# (Intercept)                  0.1008     0.1676   0.601    0.549    
# X_Temperature               -2.9809     0.1621 -18.385  < 2e-16 ***
# X_Humidity                  -4.2952     0.1844 -23.290  < 2e-16 ***
# X_Altitude                  -1.1809     0.1636  -7.216 1.48e-10 ***
# X_RestTimeFromLastMatch      7.1271     0.1727  41.263  < 2e-16 ***
# X_AvgPlayerValue             2.9759     0.1646  18.078  < 2e-16 ***
# X_SupportersImpact           2.9035     0.1608  18.059  < 2e-16 ***
# X_OpposingSupportersImpact  -2.0448     0.1693 -12.081  < 2e-16 ***
#   ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Residual standard error: 1.631 on 92 degrees of freedom
# Multiple R-squared:  0.9725,	Adjusted R-squared:  0.9705 
# F-statistic: 465.6 on 7 and 92 DF,  p-value: < 2.2e-16
# 
# [1] "==================  MSE  =================="
# [1] 2.939669

bestSubset = bestSubsets$model[[8]]
bestSubsetMSE = bestSubsets$MSE[[8]]
lm.inspect(bestSubset, 5, 5)

# [1] "================= SUMMARY ================="
# 
# Call:
#  lm(formula = f, data = data, x = TRUE, y = TRUE)
# 
# Residuals:
# Min      1Q  Median      3Q     Max 
# -3.0689 -0.9752  0.1234  0.8329  3.5193 
# 
# Coefficients:
# Estimate Std. Error t value Pr(>|t|)    
# (Intercept)                  -0.01737    0.14638  -0.119    0.906    
# X_Temperature                -3.13390    0.14270 -21.962  < 2e-16 ***
# X_Humidity                   -4.14164    0.16168 -25.617  < 2e-16 ***
# X_Altitude                   -1.31821    0.14349  -9.187 1.28e-14 ***
# X_RestTimeFromLastMatch       7.16980    0.14947  47.969  < 2e-16 ***
# X_AvgPlayerValue              3.13260    0.14493  21.614  < 2e-16 ***
# X_AvgGoalConcededLastMatches -0.88346    0.15577  -5.672 1.66e-07 ***
# X_SupportersImpact            2.85313    0.13924  20.491  < 2e-16 ***
# X_OpposingSupportersImpact   -2.14316    0.14731 -14.549  < 2e-16 ***
#   ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Residual standard error: 1.409 on 91 degrees of freedom
# Multiple R-squared:  0.9797,	Adjusted R-squared:  0.9779 
# F-statistic: 549.4 on 8 and 91 DF,  p-value: < 2.2e-16
# 
# [1] "==================  MSE  =================="
# [1] 2.190035

#Modello migliore

#============================  RIDGE E LASSO   =================================

# Add non linearities before scaling
bestInteractions = list(
  'X_SupportersImpact*X_OpposingSupportersImpact',
  'X_SupportersImpact*X_MatchRelevance',
  'I(X_AvgPlayerValue^2)'
)    

ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)
ds_scaled = addNonLinearities(ds, bestInteractions)

lambda_grid = 10^seq(4, -6, length = 2000)

models = lm.shrinkage(ds_scaled, lambda_grid, nMSE=10, folds=10, showPlot=T)
coef(models$lasso$model, s = models$lasso$bestlambda)
coef(models$ridge$model, s = models$ridge$bestlambda)

min(models$lasso$cvm)
min(models$ridge$cvm)

#Troppi predittori per poco guadagno

# predictedY = predictWithGlmnet(models$lasso, newx=as.matrix(ds_scaled[,-1]))

#============================= ELASTIC NET  ===============================

bestInteractions = list(
  'X_SupportersImpact*X_OpposingSupportersImpact',
  'X_SupportersImpact*X_MatchRelevance',
  'I(X_AvgPlayerValue^2)'
)  


ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)
ds_scaled = addNonLinearities(ds, bestInteractions)

lambda_grid = 10^seq(4, -6, length = 200)
alpha_grid = seq(0,1,length = 100)

best_mse = mean_cvMSE(bestSubset, 10, 10)

MSEs = lm.elasticNet(ds_scaled, alpha_grid, lambda_grid, nMSE=10, folds=10, best_mse = best_mse, showPlot = T, verbose = T)

lm.plotElasticNet(alpha_grid, MSEs, best_mse)

#======================= LINEAR REGRESSION - ISSUES =======================

best_model = lm(Y_PhysicalEndurance ~ X_Temperature + X_Humidity+X_Altitude+X_RestTimeFromLastMatch+X_AvgPlayerValue+
                  X_AvgGoalConcededLastMatches+X_SupportersImpact+X_OpposingSupportersImpact,data=ds, y=T,x=T)

lm.inspect(best_model, 10)

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

indices_to_be_removed = hats_indices

if(length(indices_to_be_removed) > 0) {
  ds = ds[-indices_to_be_removed,]
}
best_model = lm.refit(best_model, ds)
lm.inspect(best_model, 10, 10)

hat.plot(best_model)

#======================= CONCLUSION =======================

best_formula = "Y_PhysicalEndurance ~ X_Temperature + X_Humidity+X_Altitude+X_RestTimeFromLastMatch+X_AvgPlayerValue+
                  X_AvgGoalConcededLastMatches+X_SupportersImpact+X_OpposingSupportersImpact"

best_summary = '
              [1] "================= SUMMARY ================="
              
              Call:
              lm(formula = best_formula, data = ds, x = T, y = T)
              
              Residuals:
                  Min      1Q  Median      3Q     Max 
              -3.0689 -0.9752  0.1234  0.8329  3.5193 
              
              Coefficients:
                                           Estimate Std. Error t value Pr(>|t|)    
              (Intercept)                  -0.01737    0.14638  -0.119    0.906    
              X_Temperature                -3.13390    0.14270 -21.962  < 2e-16 ***
              X_Humidity                   -4.14164    0.16168 -25.617  < 2e-16 ***
              X_Altitude                   -1.31821    0.14349  -9.187 1.28e-14 ***
              X_RestTimeFromLastMatch       7.16980    0.14947  47.969  < 2e-16 ***
              X_AvgPlayerValue              3.13260    0.14493  21.614  < 2e-16 ***
              X_AvgGoalConcededLastMatches -0.88346    0.15577  -5.672 1.66e-07 ***
              X_SupportersImpact            2.85313    0.13924  20.491  < 2e-16 ***
              X_OpposingSupportersImpact   -2.14316    0.14731 -14.549  < 2e-16 ***
              ---
              Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
              
              Residual standard error: 1.409 on 91 degrees of freedom
              Multiple R-squared:  0.9797,	Adjusted R-squared:  0.9779 
              F-statistic: 549.4 on 8 and 91 DF,  p-value: < 2.2e-16
              
              [1] "==================  MSE  =================="
              [1] 2.242226
              '

# OTTENUTO CON BEST SUBSET

best_model = lm(best_formula, data=ds, y = T, x = T)
lm.inspect(best_model,10)
exportCOEF(best_model$coefficients)
