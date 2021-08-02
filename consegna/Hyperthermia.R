#==============================   CONFIG     ============================

ABS_PATH = 'C:/Users/rosar/OneDrive/Desktop/Clone_git/SDAgruppo2'
DATASET_FILENAME = 'RegressionData_final.csv'
Y_LABEL = 'Y_Hyperthermia'
PREDICTORS_NUMBER = 10

#================================ START =================================

setwd(ABS_PATH)
source('./utils.R')
ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)


#==================== REGRESSION WITHOUT INTERACTIONS ====================

baseModel=lm.byIndices(ds, -1)
lm.inspect(baseModel, 10)

#R^2 = 0.9719 MSE = 3.802612
# Call:
#   lm(formula = f, data = data, x = T, y = T)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -4.5720 -1.2673  0.1346  0.9982  4.4218 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)                  -0.31668    0.18723  -1.691   0.0943 .  
# X_Temperature                 7.87600    0.18364  42.888  < 2e-16 ***
#   X_Humidity                    6.10823    0.21653  28.209  < 2e-16 ***
#   X_Altitude                   -1.06471    0.18573  -5.733 1.33e-07 ***
#   X_ClimaticConditions         -2.06652    0.17428 -11.857  < 2e-16 ***
#   X_RestTimeFromLastMatch       0.12934    0.19536   0.662   0.5096    
# X_AvgPlayerValue             -0.01668    0.18820  -0.089   0.9296    
# X_MatchRelevance              0.98563    0.17558   5.613 2.22e-07 ***
#   X_AvgGoalConcededLastMatches -0.21828    0.19976  -1.093   0.2775    
# X_SupportersImpact            0.10060    0.17792   0.565   0.5732    
# X_OpposingSupportersImpact    0.13366    0.18968   0.705   0.4829    
# ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Residual standard error: 1.801 on 89 degrees of freedom
# Multiple R-squared:  0.9719,	Adjusted R-squared:  0.9687 
# F-statistic: 307.4 on 10 and 89 DF,  p-value: < 2.2e-16
# 
# [1] "==================  MSE  =================="
# [1] 3.802612


#====================== INSPECT RELATIONSHIPS ============================

showPlotsAgainstOutput(ds, 2:(PREDICTORS_NUMBER+1))



#======================== TEST RELATIONSHIPS =============================

possibleDependencies = list('I(X_ClimaticConditions^2)', 
                            'X_Temperature',
                            'X_Humidity',
                            'X_Altitude',
                            'X_MatchRelevance'
)

dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(dependencyModel, 5)
# R^2 = 0.9248 MSE = 

possibleDependencies = list('I(X_Altitude^2)', 
                            'X_Temperature',
                            'X_Humidity',
                            'X_ClimaticConditions',
                            'X_MatchRelevance'
)

dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(dependencyModel, 5)
# R^2 = 0.9615 MSE = 4.553324
# Non Buoni
#======================== INSPECT INTERACTIONS =============================

# Collect rsquared for every linear model obtained by adding every possible
# interaction between two distinct predictors to the base model.
# Set base rsquared as default value

baseRSquared = summary( lm.byIndices(ds, -1) )$r.squared
interactionMatrix = inspectInteractionMatrix(ds, default=baseRSquared, showHeatmap = T)


#====================  BEST SUBSET SELECTION WITH INTERACTIONS   ===============

# add non linearities for best subset selection
possibleRelationships = list(
  'I(X_AvgPlayerValue^2)',
  'X_RestTimeFromLastMatch*X_AvgPlayerValue',
  'X_Altitude*X_AvgPlayerValue',
  'X_Temperature*X_AvgGoalConcededLastMatches',
  'I(X_ClimaticConditions^2)'
)    

bestSubsets = bestSubsetSelection(ds, relationships=possibleRelationships, nMSE=10, folds=10, verbose=T, method="exhaustive")
ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

bestSubset = bestSubsets$model[[5]]
lm.inspect(bestSubset, 10, 10)
# [1] "================= SUMMARY ================="
# 
# Call:
#   lm(formula = f, data = data, x = TRUE, y = TRUE)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -4.7131 -1.2303  0.1376  1.0469  4.8150 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)           -0.2975     0.1811  -1.643    0.104    
# X_Temperature          7.9099     0.1778  44.477  < 2e-16 ***
#   X_Humidity             6.0863     0.2085  29.189  < 2e-16 ***
#   X_Altitude            -1.0104     0.1797  -5.624 1.91e-07 ***
#   X_ClimaticConditions  -2.0738     0.1692 -12.255  < 2e-16 ***
#   X_MatchRelevance       1.0062     0.1670   6.024 3.28e-08 ***
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Residual standard error: 1.779 on 94 degrees of freedom
# Multiple R-squared:  0.971,	Adjusted R-squared:  0.9695 
# F-statistic: 629.3 on 5 and 94 DF,  p-value: < 2.2e-16
# 
# [1] "==================  MSE  =================="
# [1] 3.346304
bestSubset = bestSubsets$model[[6]]
lm.inspect(bestSubset, 10, 10)

# [1] "================= SUMMARY ================="
# 
# Call:
#   lm(formula = f, data = data, x = TRUE, y = TRUE)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -3.7169 -1.1823  0.1194  1.0503  4.7756 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)             0.2962     0.2577   1.150  0.25329    
# X_Temperature           7.9005     0.1702  46.425  < 2e-16 ***
#   X_Humidity              6.0865     0.1995  30.509  < 2e-16 ***
#   X_Altitude             -1.0708     0.1730  -6.190 1.61e-08 ***
#   X_ClimaticConditions   -2.1421     0.1634 -13.111  < 2e-16 ***
#   X_MatchRelevance        0.9945     0.1598   6.222 1.39e-08 ***
#   I(X_AvgPlayerValue^2)  -0.5845     0.1878  -3.113  0.00246 ** 
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Residual standard error: 1.702 on 93 degrees of freedom
# Multiple R-squared:  0.9737,	Adjusted R-squared:  0.972 
# F-statistic: 574.5 on 6 and 93 DF,  p-value: < 2.2e-16
# 
# [1] "==================  MSE  =================="
# [1] 3.149905
# Il modello a 6 predittori per adesso risulta il migliore

#============================  RIDGE E LASSO   =================================

# Add non linearities before scaling
bestInteractions = list(
  'I(X_AvgPlayerValue^2)',
  'X_RestTimeFromLastMatch*X_AvgPlayerValue',
  'X_Altitude*X_AvgPlayerValue',
  'X_Temperature*X_AvgGoalConcededLastMatches',
  'I(X_ClimaticConditions^2)'
)  

ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)
ds_scaled = addNonLinearities(ds, bestInteractions)

lambda_grid = 10^seq(4, -6, length = 2000)

models = lm.shrinkage(ds_scaled, lambda_grid, nMSE=10, folds=10, showPlot=T)
coef(models$lasso$model, s = models$lasso$bestlambda)
coef(models$ridge$model, s = models$ridge$bestlambda)

min(models$ridge$cvm)
# MSE = 3.11701
min(models$lasso$cvm)
# MSE = 2.989718

# Lasso offre poco miglioramento e inserisce molti più predittori
# di best subset quindi consideriamo migliore il modello
# best subsets

#============================= ELASTIC NET  ===============================

bestInteractions = list(
  'I(X_AvgPlayerValue^2)',
  'X_RestTimeFromLastMatch*X_AvgPlayerValue',
  'X_Altitude*X_AvgPlayerValue',
  'X_Temperature*X_AvgGoalConcededLastMatches',
  'I(X_ClimaticConditions^2)'
)  

ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)
ds_scaled = addNonLinearities(ds, bestInteractions)

lambda_grid = 10^seq(4, -6, length = 500)
alpha_grid = seq(0,1,length = 100)

best_mse = mean_cvMSE(bestSubset, 10, 10)

MSEs = lm.elasticNet(ds_scaled, alpha_grid, lambda_grid, nMSE=10, folds=10, best_mse = best_mse, showPlot = T, verbose = T)

lm.plotElasticNet(alpha_grid, MSEs, best_mse)
# elastic net e il modello da noi trovato sono molto vicini
#======================= LINEAR REGRESSION - ISSUES =======================

best_model = lm.byFormulaChunks(ds, list(
  "X_Temperature",
  "X_Humidity",
  "X_Altitude",
  "X_ClimaticConditions",
  "X_MatchRelevance",
  "I(X_AvgPlayerValue^2)"
))

# 1) non-linearities & homoschedasticity ----------------------------------
# analyze residuals
plot(best_model, which=1)


# 2) high leverage points -------------------------------------------------
# # compute and plot hat values

hat.plot(best_model)

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

# Non ci sono punti di leverage e outlier

#======================= CONCLUSION =======================
# Il modello scelto lo abbiamo trovato con best subset
best_formula = "Y_Hyperthermia ~ X_Temperature + X_Humidity + X_Altitude + 
                X_ClimaticConditions + X_MatchRelevance + I(X_AvgPlayerValue^2)"
best_summary = '# [1] "================= SUMMARY ================="
# 
# Call:
#   lm(formula = f, data = data, x = TRUE, y = TRUE)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -3.7169 -1.1823  0.1194  1.0503  4.7756 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)             0.2962     0.2577   1.150  0.25329    
# X_Temperature           7.9005     0.1702  46.425  < 2e-16 ***
#   X_Humidity              6.0865     0.1995  30.509  < 2e-16 ***
#   X_Altitude             -1.0708     0.1730  -6.190 1.61e-08 ***
#   X_ClimaticConditions   -2.1421     0.1634 -13.111  < 2e-16 ***
#   X_MatchRelevance        0.9945     0.1598   6.222 1.39e-08 ***
#   I(X_AvgPlayerValue^2)  -0.5845     0.1878  -3.113  0.00246 ** 
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Residual standard error: 1.702 on 93 degrees of freedom
# Multiple R-squared:  0.9737,	Adjusted R-squared:  0.972 
# F-statistic: 574.5 on 6 and 93 DF,  p-value: < 2.2e-16
# 
# [1] "==================  MSE  =================="
# [1] 3.149905'
best_model = lm(best_formula, data=ds, y=T, x=T)
lm.inspect(best_model, 10, 10)
exportCOEF(best_model$coefficients)
