#==============================   CONFIG     ============================

ABS_PATH = 'C:/Users/rosar/OneDrive/Desktop/Clone_git/SDAgruppo2'
DATASET_FILENAME = 'RegressionData_final.csv'
Y_LABEL = 'Y_Dehydration'
PREDICTORS_NUMBER = 10

#================================ START =================================

setwd(ABS_PATH)
source('./utils.R')
ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)


#==================== REGRESSION WITHOUT INTERACTIONS ====================

baseModel=lm.byIndices(ds, -1)
lm.inspect(baseModel, 10)
#MSE = 2.317057 R^2 = 0.9672
           

#====================== INSPECT RELATIONSHIPS ============================

showPlotsAgainstOutput(ds, 2:(PREDICTORS_NUMBER+1))



#======================== TEST RELATIONSHIPS =============================

possibleDependencies = list('I(X_ClimaticConditions^2)', 
                            'X_Temperature',
                            'X_Humidity',
                            'X_Altitude',
                            'X_MatchRelevance',
                            'X_SupportersImpact'
)

dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(dependencyModel, 5)
# RSquared: 0.7505, MSE: 15.89145 


possibleDependencies = list('map_dbl(X_Temperature,mysqrt)', 
                            'X_Humidity',
                            'X_Altitude',
                            'X_MatchRelevance',
                            'X_SupportersImpact',
                            'X_ClimaticConditions'
)

dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(dependencyModel, 5)
# RSquared: 0.9448, MSE: 3.427075


#======================== INSPECT INTERACTIONS =============================

# Collect rsquared for every linear model obtained by adding every possible
# interaction between two distinct predictors to the base model.
# Set base rsquared as default value

baseRSquared = summary( lm.byIndices(ds, -1) )$r.squared
interactionMatrix = inspectInteractionMatrix(ds, default=baseRSquared, showHeatmap = T)



#====================  BEST SUBSET SELECTION WITH INTERACTIONS   ===============

# add non linearities for best subset selection
possibleRelationships = list(
  'X_Altitude*X_Temperature',
  'X_Altitude*X_SupportersImpact',
  'X_Humidity*X_SupportersImpact',
  'X_AvgGoalConcededLastMatches*X_SupportersImpact'
)    

bestSubsets = bestSubsetSelection(ds, relationships=possibleRelationships, nMSE=10, folds=5, verbose=T, method="exhaustive")
ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

bestSubset = bestSubsets$model[[5]]
# RSquared: 0.9613, MSE: 2.288689

#bestSubset = bestSubsets$model[[6]]
# RSquared: 0.9613, MSE: 2.288689
lm.inspect(bestSubset, 10, 10)
# RSquared: 0.9653, MSE: 2.15
# miglioramenti anche nei residui

# Il modello a 5 con meno predittori ha più o meno lo stesso
# MSE di quello a 6 quindi scegliamo 5
# Call:
#   lm(formula = f, data = data, x = TRUE, y = TRUE)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -2.9942 -0.9685 -0.3160  1.1070  3.5028 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)           -0.2334     0.1491  -1.566    0.121    
# X_Temperature          5.9516     0.1464  40.654  < 2e-16 ***
#   X_Humidity             3.1406     0.1716  18.297  < 2e-16 ***
#   X_Altitude             2.1466     0.1479  14.513  < 2e-16 ***
#   X_ClimaticConditions  -3.1766     0.1393 -22.805  < 2e-16 ***
#   X_MatchRelevance       0.9660     0.1375   7.026 3.33e-10 ***
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Residual standard error: 1.464 on 94 degrees of freedom
# Multiple R-squared:  0.9613,	Adjusted R-squared:  0.9593 
# F-statistic: 467.4 on 5 and 94 DF,  p-value: < 2.2e-16
# 
# [1] "==================  MSE  =================="
# [1] MSE = 2.288689


#============================  RIDGE E LASSO   =================================

# Add non linearities before scaling
bestInteractions = list(
  # 'I(X_Altitude^2)',
  # 'X_Altitude*X_Temperature',
  # 'X_Altitude*X_SupportersImpact',
  # 'X_Humidity*X_SupportersImpact',
  # 'X_AvgGoalConcededLastMatches*X_SupportersImpact'
)   

ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)
ds_scaled = addNonLinearities(ds, bestInteractions)

lambda_grid = 10^seq(4, -6, length = 10000)

models = lm.shrinkage(ds_scaled, lambda_grid, nMSE=10, folds=10, showPlot=T)
coef(models$lasso$model, s = models$lasso$bestlambda)
coef(models$ridge$model, s = models$ridge$bestlambda)

min(models$lasso$cvm)
#MSE = 2.156135
min(models$ridge$cvm)
#MSE = 2.125588

# predictedY = predictWithGlmnet(models$lasso, newx=as.matrix(ds_scaled[,-1]))
# Ridge va meglio di tutti ma aggiunge tutti i predittori
# mentre best subset ne seleziona solo 5
#============================= ELASTIC NET  ===============================

bestInteractions = list(
  # 'I(X_Altitude^2)',
  # 'X_Altitude*X_Temperature',
  # 'X_Altitude*X_SupportersImpact',
  # 'X_Humidity*X_SupportersImpact',
  # 'X_AvgGoalConcededLastMatches*X_SupportersImpact'
)  

ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)
ds_scaled = addNonLinearities(ds, bestInteractions)

lambda_grid = 10^seq(4, -6, length = 500)
alpha_grid = seq(0,1,length = 100)

best_mse = mean_cvMSE(bestSubset, 10, 10)

MSEs = lm.elasticNet(ds_scaled, alpha_grid, lambda_grid, nMSE=10, folds=10, best_mse = best_mse, showPlot = T, verbose = T)

lm.plotElasticNet(alpha_grid, MSEs, best_mse)

#======================= LINEAR REGRESSION - ISSUES =======================

best_model = lm.byFormulaChunks(ds, list(
  "X_Temperature",
  "X_Humidity",
  "X_ClimaticConditions",
  "X_Altitude",
  "X_MatchRelevance"
))


# 1) non-linearities & homoschedasticity ----------------------------------
# analyze residuals
plot(best_model, which=1)

# 2) high leverage points -------------------------------------------------
# # compute and plot hat values

hat.plot(best_model)
hats_indices = c(15)

# 4) collinearity ---------------------------------------------------------

# check any collinearity
collinearity_models = vifs.plot(ds)

# 4) outliers -------------------------------------------------------------

outlier_indices = outlier.plot(best_model)

# x) refit ------------------------------------------------------------
# Refit non fatto perchè non ci sono punti da togliere
indices_to_be_removed = hats_indices

if(length(indices_to_be_removed) > 0) {
  ds = ds[-indices_to_be_removed,]
}
best_model = lm.refit(best_model, ds)

lm.inspect(best_model, 10, 10)

hat.plot(best_model)

# Non ho dovuto rimuovere punti perchè non ci sono outlier
# e punti ad alto leverage


#======================= CONCLUSION =======================
# Il modello scelto lo abbiamo trovato con best subset
best_formula = "Y_Dehydration ~ X_Temperature + X_Humidity + X_ClimaticConditions + 
                X_Altitude + X_MatchRelevance"
best_summary = '# Call:
#   lm(formula = f, data = data, x = TRUE, y = TRUE)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -2.9942 -0.9685 -0.3160  1.1070  3.5028 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)           -0.2334     0.1491  -1.566    0.121    
# X_Temperature          5.9516     0.1464  40.654  < 2e-16 ***
#   X_Humidity             3.1406     0.1716  18.297  < 2e-16 ***
#   X_Altitude             2.1466     0.1479  14.513  < 2e-16 ***
#   X_ClimaticConditions  -3.1766     0.1393 -22.805  < 2e-16 ***
#   X_MatchRelevance       0.9660     0.1375   7.026 3.33e-10 ***
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Residual standard error: 1.464 on 94 degrees of freedom
# Multiple R-squared:  0.9613,	Adjusted R-squared:  0.9593 
# F-statistic: 467.4 on 5 and 94 DF,  p-value: < 2.2e-16
# 
# [1] "==================  MSE  =================="
# [1] MSE = 2.288689'
best_model = lm(best_formula, data=ds, y=T, x=T)
lm.inspect(best_model, 10, 10)

exportCOEF(best_model$coefficients)
