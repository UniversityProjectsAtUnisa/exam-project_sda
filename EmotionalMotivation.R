#==============================   CONFIG     ============================

ABS_PATH = 'C:/Users/gorra/Desktop/new_git/SDAgruppo2'
DATASET_FILENAME = 'RegressionData_SDA_AH_group2.csv'
Y_LABEL = 'Y_EmotionalMotivation'
PREDICTORS_NUMBER = 10

#================================ START =================================

setwd(ABS_PATH)
source('./utils.R')
ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)


#==================== REGRESSION WITHOUT INTERACTIONS ====================

bestSubsets = bestSubsetSelection(ds, relationships=NULL, nMSE=10, folds=10, verbose=T, method="exhaustive")
ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

baseModel=bestSubsets$model[[2]]
lm.inspect(baseModel, 5)            #MSE = 2.323413

#====================== INSPECT RELATIONSHIPS ============================

showPlotsAgainstOutput(ds, 2:(PREDICTORS_NUMBER+1))

#======================== TEST RELATIONSHIPS =============================

possibleDependencies = list('I(X_RestTimeFromLastMatch^2)',
                            'X_AvgPlayerValue',
                            'X_MatchRelevance')

dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(dependencyModel, 5)     

possibleDependencies = list('I(X_Altitude^2)',
                            'X_AvgPlayerValue',
                            'X_MatchRelevance')

dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(dependencyModel, 5)  


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

possibleDependencies = list('X_AvgPlayerValue',
                            'X_MatchRelevance'                            )

possibleInteractions = list('X_ClimaticConditions*X_MatchRelevance')
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(dependencyModelWithPossibleInteractions, 5)


possibleInteractions = list('X_AvgPlayerValue*X_OpposingSupportersImpact')
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(dependencyModelWithPossibleInteractions, 5)


possibleInteractions = list('X_Altitude*X_AvgGoalConcededLastMatches')
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(dependencyModelWithPossibleInteractions, 5)

possibleInteractions = list(
  'X_Altitude*X_AvgGoalConcededLastMatches', 
  'X_AvgPlayerValue*X_OpposingSupportersImpact',
  'X_ClimaticConditions*X_MatchRelevance'
)
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(dependencyModelWithPossibleInteractions, 5)


#====================  BEST SUBSET SELECTION WITH INTERACTIONS   ===============

# add non linearities for best subset selection
possibleRelationships = list('I(X_RestTimeFromLastMatch^2)',
                             'I(X_Altitude^2)',
                             'X_Altitude*X_AvgGoalConcededLastMatches'
)    

bestSubsets = bestSubsetSelection(ds, relationships=possibleRelationships, nMSE=10, folds=10, verbose=T, method="exhaustive")
ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

bestSubset = bestSubsets$model[[2]]
bestSubsetMSE = bestSubsets$MSE[[2]]
lm.inspect(bestSubset, 5, 5)

bestSubsetOSE = oneStandardErrorSubset(bestSubsets)

plot(bestSubsetOSE, 1)
plot(bestSubset,1)


#============================  RIDGE E LASSO   =================================

# Add non linearities before scaling
bestInteractions = list('I(X_RestTimeFromLastMatch^2)',
                        'I(X_Altitude^2)',
                        'X_Altitude*X_AvgGoalConcededLastMatches'
)     

ds_scaled = ds.scale(addNonLinearities(ds, bestInteractions))

lambda_grid = 10^seq(4, -6, length = 2000)

models = lm.shrinkage(ds_scaled, lambda_grid, nMSE=2, folds=4, showPlot=T)
coef(models$lasso$model, s = models$lasso$bestlambda)
coef(models$ridge$model, s = models$ridge$bestlambda)


# predictedY = predictWithGlmnet(models$lasso, newx=as.matrix(ds_scaled[,-1]))


#============================= ELASTIC NET  ===============================

bestInteractions = list('I(X_RestTimeFromLastMatch^2)',
                        'I(X_Altitude^2)',
                        'X_Altitude*X_AvgGoalConcededLastMatches'
)     

ds_scaled = ds.scale(addNonLinearities(ds, bestInteractions))

lambda_grid = 10^seq(4, -6, length = 500)
alpha_grid = seq(0,1,length = 100)


MSEs = lm.elasticNet(ds_scaled, alpha_grid, lambda_grid, nMSE=10, folds=10, best_mse = bestSubsetMSE, showPlot = T, verbose = T)

# lm.plotElasticNet(alpha_grid, MSEs, 2)

#======================= LINEAR REGRESSION - ISSUES =======================
# the best model to analyze
best_model = bestSubset # or any other (not glmnet model!)

best_model = lm(Y_EmotionalMotivation ~ X_AvgPlayerValue + X_MatchRelevance, data=ds, y=T,x=T)

# 1) non-linearities & homoschedasticity ----------------------------------
# analyze residuals
plot(best_model, which=1)
# La linea rossa non è dritta quindi c'è della non linearità che non è stata spiegata

# 2) high leverage points -------------------------------------------------
# # compute hat values
# hats <- as.data.frame(hatvalues(best_model))
# #check wether any of them is much greater than the mean (p+1)/n
# num_points = dim(ds)[1]
# ds.prettyPlot(hats/(NUM+1)*num_points, xlabel, ylabel, title)
# # non c'è nessun valore >>(p+1)/n

hats = hat.plot(best_model)
index_leverage=which.max(hats)
ds_without_high_leverage = ds[-index_leverage,]

# 4) collinearity ---------------------------------------------------------

# check any collinearity
collinearity_models = vifs.plot(ds)

# 4) outliers -------------------------------------------------------------

outlier_indices = outlier.plot(best_model)


if(length(outlier_indices) > 0) {
  ds_without_outliers = ds_without_high_leverage[-outlier_indices,]
}else{
  ds_without_outliers = ds_without_high_leverage
}

refitted_best_model = lm.refit(best_model, ds_without_outliers)

lm.inspect(refitted_best_model, 5, 5)

#plotres(models$lasso$model, s=models$lasso$bestlambda)



#======================= CONCLUSION =======================

best_formula = "Y_EmotionalMotivation ~ X_AvgPlayerValue + X_MatchRelevance"

best_summary = '
            [1] "================= SUMMARY ================="
                        
            Call:
            lm(formula = formula(model), data = data, x = T, y = T)
            
            Residuals:
                 Min       1Q   Median       3Q      Max 
            -2.62176 -0.96034 -0.08137  0.53190  2.95914 
            
            Coefficients:
                             Estimate Std. Error t value Pr(>|t|)    
            (Intercept)        1.5391     1.1919   1.291    0.205    
            X_AvgPlayerValue   4.9198     0.8729   5.636 2.13e-06 ***
            X_MatchRelevance  12.2402     2.2936   5.337 5.35e-06 ***
            ---
            Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
            
            Residual standard error: 1.426 on 36 degrees of freedom
            Multiple R-squared:  0.6457,	Adjusted R-squared:  0.626 
            F-statistic: 32.81 on 2 and 36 DF,  p-value: 7.733e-09
            
            [1] "==================  MSE  =================="
            [1] 2.385925
'

best_model = lm(best_formula, data=ds_without_outliers,y=T,x=T)
lm.inspect(best_model, 5, 5)

exportCOEF(best_model$coefficients)
