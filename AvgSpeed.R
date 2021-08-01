#==============================   CONFIG     ============================

ABS_PATH = 'C:/Users/marco/Documents/UNISA/SDA/progetto/SDAgruppo2'
DATASET_FILENAME = 'RegressionData_SDA_AH_group2.csv'
Y_LABEL = 'Y_AvgSpeed'
PREDICTORS_NUMBER = 10

#================================ START =================================

setwd(ABS_PATH)
source('./utils.R')
ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)


#==================== REGRESSION WITHOUT INTERACTIONS ====================

baseModel=lm.byIndices(ds, -1)
lm.inspect(baseModel, 5)


#====================== INSPECT RELATIONSHIPS ============================

showPlotsAgainstOutput(ds, 2:(PREDICTORS_NUMBER+1))



#======================== TEST RELATIONSHIPS =============================

possibleDependencies = list('X_Temperature', 
                            'I(X_Temperature^2)',
                            'X_ClimaticConditions',
                            'X_RestTimeFromLastMatch',
                            'X_AvgPlayerValue',
                            'X_SupportersImpact',
                            'I(X_SupportersImpact^2)',            
                            'X_OpposingSupportersImpact',            
                            'I(X_OpposingSupportersImpact^2)'            
                            )
dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(dependencyModel, 5)
# RSquared: 0.8482, MSE: 2.999463


possibleDependencies = list('X_Temperature', 
                            'I(X_Temperature^2)',
                            'X_RestTimeFromLastMatch',
                            'X_AvgPlayerValue')
dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(dependencyModel, 5)
# RSquared: 0.8249, MSE: 2.381227


possibleDependencies = list('I(X_Temperature^2)',
                            'X_RestTimeFromLastMatch',
                            'X_AvgPlayerValue')
dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(dependencyModel, 5)
# RSquared: 0.8242, MSE: 2.251368


possibleDependencies = list('I(X_Temperature^2)',
                            'X_RestTimeFromLastMatch',
                            'X_AvgPlayerValue',
                            'I(X_ClimaticConditions^2)')
dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(dependencyModel, 5)
# RSquared: 0.8452, MSE: 2.169601 # BEST


possibleDependencies = list('X_Temperature',  
                            'I(X_Temperature^2)',
                            'X_RestTimeFromLastMatch',
                            'X_AvgPlayerValue',
                            'I(X_AvgPlayerValue^2)')
dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(dependencyModel, 5)
# RSquared: 0.8279, MSE: 2.55414


possibleDependencies = list('I(X_Temperature^2)',
                            'X_RestTimeFromLastMatch',
                            'X_AvgPlayerValue',
                            'I(X_AvgPlayerValue^2)')
dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(dependencyModel, 5)
# RSquared: 0.8267, MSE: 2.439745


possibleDependencies = list('I(X_Temperature^2)',
                            'X_RestTimeFromLastMatch',
                            'I(X_AvgPlayerValue^2)')
dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(dependencyModel, 5)
# RSquared: 0.8197, MSE: 2.588137


#======================== INSPECT INTERACTIONS =============================

# Collect rsquared for every linear model obtained by adding every possible
# interaction between two distinct predictors to the base model.
# Set base rsquared as default value

baseRSquared = summary( lm.byIndices(ds, -1) )$r.squared
interactionMatrix = inspectInteractionMatrix(ds, default=baseRSquared, showHeatmap = T)



#========================  TEST INTERACTIONS   =============================

possibleDependencies = list('I(X_Temperature^2)',
                            'X_RestTimeFromLastMatch',
                            'X_AvgPlayerValue',
                            'I(X_ClimaticConditions^2)')



possibleInteractions = list('X_RestTimeFromLastMatch*X_AvgPlayerValue')
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(dependencyModelWithPossibleInteractions, 5)
# RSquared: 0.8646, MSE: 2.009099 # BEST


possibleInteractions = list('X_Temperature*X_Altitude',
                            'X_RestTimeFromLastMatch*X_AvgPlayerValue')
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(dependencyModelWithPossibleInteractions, 5)
# RSquared: 0.8823, MSE: 2.049297


possibleInteractions = list('X_Temperature*X_Altitude')
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(dependencyModelWithPossibleInteractions, 5)
# RSquared: 0.8686, MSE: 2.213753

#====================  BEST SUBSET SELECTION WITH INTERACTIONS   ===============

# add non linearities for best subset selection
possibleRelationships = list(
  'I(X_ClimaticConditions^2)',
  'X_RestTimeFromLastMatch*X_AvgPlayerValue', # IMPORTANTE
  'X_Temperature*X_ClimaticConditions',
  'X_Humidity*X_ClimaticConditions',
  'X_Altitude*X_ClimaticConditions',
  'X_Altitude*X_Temperature',
  'X_Altitude*X_AvgPlayerValue',
  'X_AvgPlayerValue*X_SupportersImpact'
)    

bestSubsets = bestSubsetSelection(ds, relationships=possibleRelationships, nMSE=10, folds=5, verbose=T, method="exhaustive")
ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

bestSubset = bestSubsets$model[[which.min(bestSubsets$MSE)]]

lm.inspect(bestSubset, 10, 10)


#=============  BEST SUBSETS FOR SELECTED NUMBER OF PREDICTORS   ===============

# add non linearities for best subset selection
possibleRelationships = list(
  'I(X_ClimaticConditions^2)',
  'X_RestTimeFromLastMatch*X_AvgPlayerValue', # IMPORTANTE
  'X_Temperature*X_ClimaticConditions',
  'X_Humidity*X_ClimaticConditions',
  'X_Altitude*X_ClimaticConditions',
  'X_Altitude*X_Temperature',
  'X_Altitude*X_AvgPlayerValue',
  'X_AvgPlayerValue*X_SupportersImpact'
)    

N_PREDICTORS_TO_INSPECT = 5
bestSubsets = bestSubsetsByPredictorsNumber(ds, relationships=possibleRelationships, nMSE=10, folds=5, nPredictors=N_PREDICTORS_TO_INSPECT, nSubsets=10, verbose=T)
ds.prettyPlot(bestSubsets$MSE, xdata=unlist(map(bestSubsets$model, function(model) summary(model)$r.squared)), xlab="Rank", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

bestSubset = bestSubsets$model[[which.min(bestSubsets$MSE)]]

lm.inspect(bestSubset, 10, 10)

#   BEST MODEL with less coefficients
# 
# Estimate Std. Error t value Pr(>|t|)    
# (Intercept)                      1.1032     1.3973   0.790 0.435270    
# X_RestTimeFromLastMatch          3.8572     0.6023   6.404 2.58e-07 ***
#   X_MatchRelevance                11.4692     1.8258   6.282 3.72e-07 ***
#   X_Temperature                   -5.0280     1.4540  -3.458 0.001482 ** 
#   X_AvgPlayerValue                 4.5206     1.5966   2.831 0.007731 ** 
#   X_Temperature:X_AvgPlayerValue   9.1283     2.4448   3.734 0.000689 ***


#=================== FORWARD SELECTION WITH INTERACTIONS =======================


possibleRelationships = list(
  'I(X_ClimaticConditions^2)'
)
bestSubsets = bestSubsetSelection(ds, relationships=possibleRelationships, nMSE=10, folds=5, method="forward", nvmax=8, verbose=T)
bestSubset = bestSubsets$model[[which.min(bestSubsets$MSE)]]

ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")


#============================  RIDGE E LASSO   =================================

# Add non linearities before scaling
bestInteractions = list(
  'I(X_ClimaticConditions^2)',
  'X_RestTimeFromLastMatch*X_AvgPlayerValue', # IMPORTANTE
  'X_Temperature*X_ClimaticConditions',
  'X_Humidity*X_ClimaticConditions',
  'X_Altitude*X_ClimaticConditions',
  'X_Altitude*X_Temperature',
  'X_Altitude*X_AvgPlayerValue',
  'X_AvgPlayerValue*X_SupportersImpact'
)    
ds_scaled = ds.scale(addNonLinearities(ds, bestInteractions))

lambda_grid = 10^seq(4, -6, length = 10000)

models = lm.shrinkage(ds_scaled, lambda_grid, nMSE=10, folds=10, showPlot=T)
coef(models$lasso$model, s = models$lasso$bestlambda)
coef(models$ridge$model, s = models$ridge$bestlambda)


# predictedY = predictWithGlmnet(models$lasso, newx=as.matrix(ds_scaled[,-1]))

#============================= ELASTIC NET  ===============================

bestInteractions = list(
  'I(X_ClimaticConditions^2)',
  'X_RestTimeFromLastMatch*X_AvgPlayerValue', # IMPORTANTE
  'X_Temperature*X_ClimaticConditions',
  'X_Humidity*X_ClimaticConditions',
  'X_Altitude*X_ClimaticConditions',
  'X_Altitude*X_Temperature',
  'X_Altitude*X_AvgPlayerValue',
  'X_AvgPlayerValue*X_SupportersImpact'
)    
ds_scaled = ds.scale(addNonLinearities(ds, bestInteractions))

lambda_grid = 10^seq(4, -6, length = 2000)
alpha_grid = seq(0,1,length = 100)

best_mse = mean_cvMSE(bestSubset, 10, 10)

MSEs = lm.elasticNet(ds_scaled, alpha_grid, lambda_grid, nMSE=10, folds=10, best_mse = best_mse, showPlot = T, verbose = T)

lm.plotElasticNet(alpha_grid, MSEs, best_mse)

#======================= LINEAR REGRESSION - ISSUES =======================

best_model = lm.byFormulaChunks(ds, list(
    "X_Temperature",
    "I(X_ClimaticConditions^2)",
    "X_RestTimeFromLastMatch",
    "X_AvgPlayerValue",
    "X_RestTimeFromLastMatch:X_AvgPlayerValue"
))

# the best model to analyze
best_model = bestSubset # or any other (not glmnet model!)

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
outlier_indices = c(3)

# x) refit ------------------------------------------------------------

indices_to_be_removed = c(40)

if(length(indices_to_be_removed) > 0) {
  ds = ds[-indices_to_be_removed,]
}
best_model = lm.refit(best_model, ds)

lm.inspect(best_model, 5, 5)


#======================= CONCLUSION =======================

best_formula = "Y_AvgSpeed ~ X_Temperature + I(X_ClimaticConditions^2) + X_RestTimeFromLastMatch + 
    X_AvgPlayerValue + X_RestTimeFromLastMatch:X_AvgPlayerValue"
best_summary = '
[1] "================= SUMMARY ================="

Call:
lm(formula = formula(model), data = data, x = T, y = T)

Residuals:
    Min      1Q  Median      3Q     Max 
-2.8045 -0.7973  0.1838  0.8092  2.5491 

Coefficients:
                                         Estimate Std. Error t value Pr(>|t|)    
(Intercept)                               -2.6659     1.4545  -1.833   0.0759 .  
X_Temperature                             -4.1141     0.8142  -5.053 1.58e-05 ***
I(X_ClimaticConditions^2)                 -6.2975     2.4425  -2.578   0.0146 *  
X_RestTimeFromLastMatch                    9.0207     1.6566   5.445 4.95e-06 ***
X_AvgPlayerValue                          10.4910     1.9834   5.289 7.85e-06 ***
X_RestTimeFromLastMatch:X_AvgPlayerValue  -6.6418     2.7427  -2.422   0.0211 *  
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 1.33 on 33 degrees of freedom
Multiple R-squared:  0.8299,	Adjusted R-squared:  0.8042 
F-statistic: 32.21 on 5 and 33 DF,  p-value: 8.74e-12

[1] "==================  MSE  =================="
[1] 2.043521
'
best_model = lm(best_formula, data=ds)
exportCOEF(best_model$coefficients)
