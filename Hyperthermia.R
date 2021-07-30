#==============================   CONFIG     ============================

ABS_PATH = 'C:/Users/gorra/Desktop/new_git/SDAgruppo2'
DATASET_FILENAME = 'RegressionData_SDA_AH_group2.csv'
Y_LABEL = 'Y_Hyperthermia'
PREDICTORS_NUMBER = 10

#================================ START =================================

setwd(ABS_PATH)
source('./utils.R')
ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)


#==================== REGRESSION WITHOUT INTERACTIONS ====================

bestSubsets = bestSubsetSelection(ds, relationships=NULL, nMSE=10, folds=10, verbose=T, method="exhaustive")
ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

baseModel=bestSubsets$model[[4]]
lm.inspect(baseModel, 10, 10)

#   [1] "================= SUMMARY ================="
# 
#   Call:
#   lm(formula = f, data = data, x = TRUE, y = TRUE)
# 
#   Residuals:
#   Min      1Q  Median      3Q     Max 
#   -2.5895 -0.8288  0.0764  0.9286  3.3492 
# 
#   Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
#   (Intercept)                 -3.6123     1.7753  -2.035 0.049506 *  
#   X_Temperature                7.6024     0.8205   9.266    6e-11 ***
#   X_Humidity                   4.0482     1.0138   3.993 0.000319 ***
#   X_AvgPlayerValue             2.0764     0.9871   2.103 0.042682 *  
#   X_OpposingSupportersImpact   5.0944     2.4478   2.081 0.044790 *  
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
#   Residual standard error: 1.402 on 35 degrees of freedom
#   Multiple R-squared:  0.7359,	Adjusted R-squared:  0.7057 
#   F-statistic: 24.38 on 4 and 35 DF,  p-value: 1.058e-09
# 
#   [1] "==================  MSE  =================="
#   [1] 2.174403


#====================== INSPECT RELATIONSHIPS ============================

showPlotsAgainstOutput(ds, 2:(PREDICTORS_NUMBER+1))



#======================== TEST RELATIONSHIPS =============================


# L'ANDAMENTO DEI RESIDUI DIMOSTRA CHE NON SONO PRESENTI NON LINEARITÀ


#======================== INSPECT INTERACTIONS =============================

# Collect rsquared for every linear model obtained by adding every possible
# interaction between two distinct predictors to the base model.
# Set base rsquared as default value

baseRSquared = summary( lm.byIndices(ds, -1) )$r.squared
interactionMatrix = inspectInteractionMatrix(ds, default=baseRSquared, showHeatmap = T)



#========================  TEST INTERACTIONS   =============================

possibleDependencies = list('X_Temperature',
                            'X_Humidity',       
                            'X_AvgPlayerValue',  
                            'X_OpposingSupportersImpact'
)

possibleInteractions = list('X_OpposingSupportersImpact*X_Temperature')
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(dependencyModelWithPossibleInteractions, 5)
# RSquared: 0.7425, MSE: 2.2 


possibleInteractions = list('X_RestTimeFromLastMatch*X_OpposingSupportersImpact')
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(dependencyModelWithPossibleInteractions, 5)
# RSquared: 0.7623, MSE: 2.396446


possibleInteractions = list('X_Humidity*X_OpposingSupportersImpact')
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(dependencyModelWithPossibleInteractions, 5)
# RSquared: 0.7437, MSE: 2.35


#====================  BEST SUBSET SELECTION WITH INTERACTIONS   ===============

# add non linearities for best subset selection
possibleRelationships = list(
  'X_Humidity*X_OpposingSupportersImpact',
  'X_RestTimeFromLastMatch*X_OpposingSupportersImpact',
  'X_OpposingSupportersImpact*X_Temperature',
  'X_Altitude*X_RestTimeFromLastMatch',
  'X_AvgPlayerValue*X_AvgGoalConcededLastMatches'
)    

bestSubsets = bestSubsetSelection(ds, relationships=possibleRelationships, nMSE=10, folds=10, verbose=T, method="exhaustive")
ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

bestSubset = bestSubsets$model[[4]]
# Si riconferma il modelloa quattro predittori

lm.inspect(bestSubset, 10, 10)

# [1] "================= SUMMARY ================="
# 
# Call:
#   lm(formula = f, data = data, x = TRUE, y = TRUE)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -2.5895 -0.8288  0.0764  0.9286  3.3492 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)                 -3.6123     1.7753  -2.035 0.049506 *  
#   X_Temperature                7.6024     0.8205   9.266    6e-11 ***
#   X_Humidity                   4.0482     1.0138   3.993 0.000319 ***
#   X_AvgPlayerValue             2.0764     0.9871   2.103 0.042682 *  
#   X_OpposingSupportersImpact   5.0944     2.4478   2.081 0.044790 *  
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Residual standard error: 1.402 on 35 degrees of freedom
# Multiple R-squared:  0.7359,	Adjusted R-squared:  0.7057 
# F-statistic: 24.38 on 4 and 35 DF,  p-value: 1.058e-09
# 
# [1] "==================  MSE  =================="
# [1] 2.176251


#=============  BEST SUBSETS FOR SELECTED NUMBER OF PREDICTORS   ===============

# add non linearities for best subset selection
possibleRelationships = list(
  'X_Humidity*X_OpposingSupportersImpact',
  'X_RestTimeFromLastMatch*X_OpposingSupportersImpact',
  'X_OpposingSupportersImpact*X_Temperature',
  'X_Altitude*X_RestTimeFromLastMatch',
  'X_AvgPlayerValue*X_AvgGoalConcededLastMatches'
)    


N_PREDICTORS_TO_INSPECT = 5
bestSubsets = bestSubsetsByPredictorsNumber(ds, relationships=possibleRelationships, nMSE=10, folds=10, nPredictors=N_PREDICTORS_TO_INSPECT, nSubsets=10, verbose=T)
ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

bestSubset = bestSubsets$model[[1]]

lm.inspect(bestSubset, 10, 10)


#============================  RIDGE E LASSO   =================================

# Add non linearities before scaling
bestInteractions = list(
  'X_Humidity*X_OpposingSupportersImpact',
  'X_RestTimeFromLastMatch*X_OpposingSupportersImpact',
  'X_OpposingSupportersImpact*X_Temperature',
  'X_Altitude*X_RestTimeFromLastMatch',
  'X_AvgPlayerValue*X_AvgGoalConcededLastMatches'
)  

ds_scaled = ds.scale(addNonLinearities(ds, bestInteractions))

lambda_grid = 10^seq(4, -6, length = 10000)

models = lm.shrinkage(ds_scaled, lambda_grid, nMSE=10, folds=10, showPlot=T)
coef(models$lasso$model, s = models$lasso$bestlambda)
coef(models$ridge$model, s = models$ridge$bestlambda)


# predictedY = predictWithGlmnet(models$lasso, newx=as.matrix(ds_scaled[,-1]))

#============================= ELASTIC NET  ===============================

bestInteractions = list(
  'X_Humidity*X_OpposingSupportersImpact',
  'X_RestTimeFromLastMatch*X_OpposingSupportersImpact',
  'X_OpposingSupportersImpact*X_Temperature',
  'X_Altitude*X_RestTimeFromLastMatch',
  'X_AvgPlayerValue*X_AvgGoalConcededLastMatches'
)  

ds_scaled = ds.scale(addNonLinearities(ds, bestInteractions))

lambda_grid = 10^seq(4, -6, length = 200)
alpha_grid = seq(0,1,length = 100)

best_mse = mean_cvMSE(bestSubset, 10, 10)

MSEs = lm.elasticNet(ds_scaled, alpha_grid, lambda_grid, nMSE=10, folds=10, best_mse = best_mse, showPlot = T, verbose = T)

lm.plotElasticNet(alpha_grid, MSEs, best_mse)

#======================= LINEAR REGRESSION - ISSUES =======================

best_model = lm.byFormulaChunks(ds, list(
  "X_Temperature",
  "X_Humidity",
  "X_AvgPlayerValue",
  "X_OpposingSupportersImpact"
))

# 1) non-linearities & homoschedasticity ----------------------------------
# analyze residuals
plot(best_model, which=1)
# La linea rossa è dritta quindi non c'è della non linearità

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



#======================= CONCLUSION =======================

best_formula = "Y_Hyperthermia ~ X_Temperature + X_Humidity + X_AvgPlayerValue + 
                X_OpposingSupportersImpact"
best_summary = '
                [1] "================= SUMMARY ================="
                
                Call:
                lm(formula = paste(utils.Y_LABEL, " ~ ", paste(chunks, collapse = " + ")), 
                    data = ds, x = T, y = T)
                
                Residuals:
                    Min      1Q  Median      3Q     Max 
                -2.5895 -0.8288  0.0764  0.9286  3.3492 
                
                Coefficients:
                                           Estimate Std. Error t value Pr(>|t|)    
                (Intercept)                 -3.6123     1.7753  -2.035 0.049506 *  
                X_Temperature                7.6024     0.8205   9.266    6e-11 ***
                X_Humidity                   4.0482     1.0138   3.993 0.000319 ***
                X_AvgPlayerValue             2.0764     0.9871   2.103 0.042682 *  
                X_OpposingSupportersImpact   5.0944     2.4478   2.081 0.044790 *  
                ---
                Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
                
                Residual standard error: 1.402 on 35 degrees of freedom
                Multiple R-squared:  0.7359,	Adjusted R-squared:  0.7057 
                F-statistic: 24.38 on 4 and 35 DF,  p-value: 1.058e-09
                
                [1] "==================  MSE  =================="
                [1] 2.2
'
best_model = lm(best_formula, data=ds, y=T, x=T)
lm.inspect(best_model, 10, 10)
