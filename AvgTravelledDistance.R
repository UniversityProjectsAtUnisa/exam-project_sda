#==============================   CONFIG     ============================

ABS_PATH = 'C:/Users/gorra/Desktop/new_git/SDAgruppo2'
DATASET_FILENAME = 'RegressionData_SDA_AH_group2.csv'
Y_LABEL = 'Y_AvgTravelledDistance'
PREDICTORS_NUMBER = 10

#================================ START =================================

setwd(ABS_PATH)
source('./utils.R')
ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)

#==================== REGRESSION WITHOUT INTERACTIONS ====================

bestSubsets = bestSubsetSelection(ds, relationships=NULL, nMSE=10, folds=10, verbose=T, method="exhaustive")
ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

baseModel=bestSubsets$model[[4]]
lm.inspect(baseModel, 5)            #MSE = 3.175712

#====================== INSPECT RELATIONSHIPS ============================

showPlotsAgainstOutput(ds, 2:(PREDICTORS_NUMBER+1))

#======================== TEST RELATIONSHIPS =============================

possibleDependencies = list('I(X_Temperature^2)',    
                            'I(X_AvgPlayerValue^3)', 
                            'X_RestTimeFromLastMatch',
                            'X_Humidity')

dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(dependencyModel, 5)       #MSE = 3.201141


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

possibleDependencies = list('I(X_Temperature^2)',    
                            'I(X_AvgPlayerValue^3)', 
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
  'X_RestTimeFromLastMatch*X_MatchRelevance', 
  'X_RestTimeFromLastMatch*X_ClimaticConditions',
  'I(X_Temperature^2)',    
  'I(X_AvgPlayerValue^3)'
)    

bestSubsets = bestSubsetSelection(ds, relationships=possibleRelationships, nMSE=10, folds=10, verbose=T, method="exhaustive")
ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

bestSubset = bestSubsets$model[[4]]
bestSubsetMSE = bestSubsets$MSE[[4]]
bestSubsetOSE = oneStandardErrorSubset(bestSubsets)

plot(bestSubsetOSE, 1)
plot(bestSubset,1)


#============================  RIDGE E LASSO   =================================

# Add non linearities before scaling
bestInteractions = list(
  'X_RestTimeFromLastMatch*X_MatchRelevance', 
  'X_RestTimeFromLastMatch*X_ClimaticConditions',
  'I(X_Temperature^2)',    
  'I(X_AvgPlayerValue^3)'
)    

ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)
ds_scaled = addNonLinearities(ds, bestInteractions)

lambda_grid = 10^seq(4, -6, length = 2000)

models = lm.shrinkage(ds_scaled, lambda_grid, nMSE=2, folds=4, showPlot=T)
coef(models$lasso$model, s = models$lasso$bestlambda)
coef(models$ridge$model, s = models$ridge$bestlambda)


# predictedY = predictWithGlmnet(models$lasso, newx=as.matrix(ds_scaled[,-1]))


#============================= ELASTIC NET  ===============================

bestInteractions = list(
  'X_RestTimeFromLastMatch*X_MatchRelevance', 
  'X_RestTimeFromLastMatch*X_ClimaticConditions',
  'I(X_Temperature^2)',    
  'I(X_AvgPlayerValue^3)'
)     

ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)
ds_scaled = addNonLinearities(ds, bestInteractions)

lambda_grid = 10^seq(4, -6, length = 500)
alpha_grid = seq(0,1,length = 100)


MSEs = lm.elasticNet(ds_scaled, alpha_grid, lambda_grid, nMSE=10, folds=10, best_mse = bestSubsetMSE, showPlot = T, verbose = T)

# lm.plotElasticNet(alpha_grid, MSEs, 2)

#======================= LINEAR REGRESSION - ISSUES =======================
# the best model to analyze
best_model = lm(Y_AvgTravelledDistance ~ X_Humidity + X_RestTimeFromLastMatch + 
                  I(X_AvgPlayerValue^3) + I(X_Temperature^2), data=ds,y=T,x=T)


# 1) non-linearities & homoschedasticity ----------------------------------
# analyze residuals
plot(best_model, which=1)
# La linea rossa non � dritta quindi c'� della non linearit� che non � stata spiegata

# 2) high leverage points -------------------------------------------------
# # compute hat values
# hats <- as.data.frame(hatvalues(best_model))
# #check wether any of them is much greater than the mean (p+1)/n
# num_points = dim(ds)[1]
# ds.prettyPlot(hats/(NUM+1)*num_points, xlabel, ylabel, title)
# # non c'� nessun valore >>(p+1)/n

hat.plot(best_model)

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

#plotres(models$lasso$model, s=models$lasso$bestlambda)


#======================= CONCLUSION =======================

best_formula = "Y_AvgTravelledDistance ~ X_Humidity + X_RestTimeFromLastMatch + 
                I(X_AvgPlayerValue^3) + I(X_Temperature^2)"

best_summary = '
                [1] "================= SUMMARY ================="
                
                Call:
                lm(formula = best_formula, data = ds_without_outliers, x = T, 
                    y = T)
                
                Residuals:
                     Min       1Q   Median       3Q      Max 
                -2.36430 -1.06620 -0.04062  0.86571  2.51945 
                
                Coefficients:
                                        Estimate Std. Error t value Pr(>|t|)    
                (Intercept)              -0.3293     0.8376  -0.393  0.69669    
                X_Humidity               -5.5659     0.9754  -5.706 2.07e-06 ***
                X_RestTimeFromLastMatch   2.0298     0.6869   2.955  0.00565 ** 
                I(X_AvgPlayerValue^3)     2.9249     0.9442   3.098  0.00389 ** 
                I(X_Temperature^2)       -6.4355     0.8108  -7.938 3.02e-09 ***
                ---
                Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
                
                Residual standard error: 1.344 on 34 degrees of freedom
                Multiple R-squared:  0.8212,	Adjusted R-squared:  0.8001 
                F-statistic: 39.03 on 4 and 34 DF,  p-value: 2.925e-12
                
                [1] "==================  MSE  =================="
                [1] 2.05679
'
best_model = lm(best_formula, data=ds_without_outliers,y=T,x=T)
lm.inspect(best_model, 5, 5)
exportCOEF(best_model$coefficients)
