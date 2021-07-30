#==============================   CONFIG     ============================

ABS_PATH = 'C:/Users/gorra/Desktop/new_git/SDAgruppo2'
DATASET_FILENAME = 'RegressionData_SDA_AH_group2.csv'
Y_LABEL = 'Y_Dehydration'
PREDICTORS_NUMBER = 10

#================================ START =================================

setwd(ABS_PATH)
source('./utils.R')
ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)


#==================== REGRESSION WITHOUT INTERACTIONS ====================

bestSubsets = bestSubsetSelection(ds, relationships=NULL, nMSE=10, folds=10, verbose=T, method="exhaustive")
ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

baseModel=bestSubsets$model[[2]]
lm.inspect(baseModel, 5)            #MSE = 2.24

#====================== INSPECT RELATIONSHIPS ============================

showPlotsAgainstOutput(ds, 2:(PREDICTORS_NUMBER+1))



#======================== TEST RELATIONSHIPS =============================

possibleDependencies = list('X_Temperature', 
                            'I(X_Altitude^2)'
)

dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(dependencyModel, 5)
# RSquared: 0.6427, MSE: 2.23 


possibleDependencies = list('map_dbl(X_Temperature,mysqrt)', 
                            'X_Altitude'
)

dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(dependencyModel, 5)
# RSquared: 0.5906, MSE: 2.54164


#======================== INSPECT INTERACTIONS =============================

# Collect rsquared for every linear model obtained by adding every possible
# interaction between two distinct predictors to the base model.
# Set base rsquared as default value

baseRSquared = summary( lm.byIndices(ds, -1) )$r.squared
interactionMatrix = inspectInteractionMatrix(ds, default=baseRSquared, showHeatmap = T)



#========================  TEST INTERACTIONS   =============================

possibleDependencies = list('X_Temperature', 
                            'I(X_Altitude^2)'
)

possibleInteractions = list('X_OpposingSupportersImpact*X_SupportersImpact')
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(dependencyModelWithPossibleInteractions, 5)
# RSquared: 0.7125, MSE: 2.106 # BEST


possibleInteractions = list('X_RestTimeFromLastMatch*X_MatchRelevance')
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(dependencyModelWithPossibleInteractions, 5)
# RSquared: 0.677, MSE: 2.8


possibleInteractions = list('X_Humidity*X_RestTimeFromLastMatch')
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(dependencyModelWithPossibleInteractions, 5)
# RSquared: 0.6916, MSE: 2.4


#====================  BEST SUBSET SELECTION WITH INTERACTIONS   ===============

# add non linearities for best subset selection
possibleRelationships = list(
  'I(X_Altitude^2)',
  'X_OpposingSupportersImpact*X_SupportersImpact',
  'X_RestTimeFromLastMatch*X_MatchRelevance',
  'X_Humidity*X_RestTimeFromLastMatch'
)    

bestSubsets = bestSubsetSelection(ds, relationships=possibleRelationships, nMSE=10, folds=5, verbose=T, method="exhaustive")
ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

bestSubset = bestSubsets$model[[8]]

lm.inspect(bestSubset, 10, 10)
# RSquared: 0.7398, MSE: 1.88
# miglioramenti anche nei residui

# Call:
#   lm(formula = f, data = data, x = TRUE, y = TRUE)
# 
# Residuals:
#   Min       1Q   Median       3Q      Max 
# -3.00781 -0.63989  0.08523  0.74946  2.41277 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)                                    21.2881     7.2755   2.926 0.006171 ** 
#   X_Temperature                                   5.8878     0.8040   7.323 2.09e-08 ***
#   X_Humidity                                      1.7595     0.9454   1.861 0.071662 .  
#   X_SupportersImpact                            -42.0206    13.9561  -3.011 0.004966 ** 
#   I(X_Altitude^2)                                 2.9356     0.7198   4.078 0.000269 ***
#   X_OpposingSupportersImpact                    -42.1432    14.9996  -2.810 0.008274 ** 
#   X_SupportersImpact:X_OpposingSupportersImpact  82.1127    28.1715   2.915 0.006350 ** 
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Residual standard error: 1.287 on 33 degrees of freedom
# Multiple R-squared:  0.7398,	Adjusted R-squared:  0.6925 
# F-statistic: 15.64 on 6 and 33 DF,  p-value: 2.074e-08


#=============  BEST SUBSETS FOR SELECTED NUMBER OF PREDICTORS   ===============

# add non linearities for best subset selection
possibleRelationships = list(
  'I(X_Altitude^2)',
  'X_OpposingSupportersImpact*X_SupportersImpact',
  'X_RestTimeFromLastMatch*X_MatchRelevance',
  'X_Humidity*X_RestTimeFromLastMatch'
)   

N_PREDICTORS_TO_INSPECT = 6
bestSubsets = bestSubsetsByPredictorsNumber(ds, relationships=possibleRelationships, nMSE=10, folds=5, nPredictors=N_PREDICTORS_TO_INSPECT, nSubsets=10, verbose=T)
ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

bestSubset = bestSubsets$model[[which.min(bestSubsets$MSE)]]

lm.inspect(bestSubset, 10, 10)


#============================  RIDGE E LASSO   =================================

# Add non linearities before scaling
bestInteractions = list(
  'I(X_Altitude^2)',
  'X_OpposingSupportersImpact*X_SupportersImpact',
  'X_RestTimeFromLastMatch*X_MatchRelevance',
  'X_Humidity*X_RestTimeFromLastMatch',
  'X_AvgPlayerValue*X_RestTimeFromLastMatch',
  'X_Temperature*X_AvgGoalConcededLastMatches'
)   

ds_scaled = ds.scale(addNonLinearities(ds, bestInteractions))

lambda_grid = 10^seq(4, -6, length = 10000)

models = lm.shrinkage(ds_scaled, lambda_grid, nMSE=10, folds=10, showPlot=T)
coef(models$lasso$model, s = models$lasso$bestlambda)
coef(models$ridge$model, s = models$ridge$bestlambda)


# predictedY = predictWithGlmnet(models$lasso, newx=as.matrix(ds_scaled[,-1]))

#============================= ELASTIC NET  ===============================

bestInteractions = list(
  'I(X_Altitude^2)',
  'X_OpposingSupportersImpact*X_SupportersImpact',
  'X_RestTimeFromLastMatch*X_MatchRelevance',
  'X_Humidity*X_RestTimeFromLastMatch',
  'X_AvgPlayerValue*X_RestTimeFromLastMatch',
  'X_Temperature*X_AvgGoalConcededLastMatches'
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
  "X_SupportersImpact",
  "I(X_Altitude^2)",
  "X_OpposingSupportersImpact",
  "X_SupportersImpact:X_OpposingSupportersImpact"
))

# 1) non-linearities & homoschedasticity ----------------------------------
# analyze residuals
plot(best_model, which=1)
# La linea rossa non è dritta quindi c'è della non linearità che non è stata spiegata

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

indices_to_be_removed = hats_indices

if(length(indices_to_be_removed) > 0) {
  ds = ds[-indices_to_be_removed,]
}
best_model = lm.refit(best_model, ds)

lm.inspect(best_model, 10, 10)

hat.plot(best_model)




#======================= CONCLUSION =======================

best_formula = "Y_Dehydration ~ X_Temperature + X_Humidity + X_SupportersImpact + 
                I(X_Altitude^2) + X_OpposingSupportersImpact + 
                X_SupportersImpact:X_OpposingSupportersImpact"
best_summary = '
            [1] "================= SUMMARY ================="
            
            Call:
            lm(formula = formula(model), data = data, x = T, y = T)
            
            Residuals:
                Min      1Q  Median      3Q     Max 
            -3.0046 -0.6527  0.1102  0.7857  2.3939 
            
            Coefficients:
                                                          Estimate Std. Error t value Pr(>|t|)    
            (Intercept)                                    20.4310     8.3035   2.461 0.019455 *  
            X_Temperature                                   5.8643     0.8224   7.131 4.31e-08 ***
            X_Humidity                                      1.7128     0.9814   1.745 0.090532 .  
            X_SupportersImpact                            -40.0249    16.6992  -2.397 0.022551 *  
            I(X_Altitude^2)                                 2.9324     0.7306   4.014 0.000337 ***
            X_OpposingSupportersImpact                    -40.4218    17.0271  -2.374 0.023767 *  
            X_SupportersImpact:X_OpposingSupportersImpact  78.2959    33.2211   2.357 0.024718 *  
            ---
            Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
            
            Residual standard error: 1.306 on 32 degrees of freedom
            Multiple R-squared:  0.7381,	Adjusted R-squared:  0.6889 
            F-statistic: 15.03 on 6 and 32 DF,  p-value: 4.268e-08


[1] "==================  MSE  =================="
[1] 2.0
'
best_model = lm(best_formula, data=ds, y=T, x=T)
lm.inspect(best_model, 10, 10)
