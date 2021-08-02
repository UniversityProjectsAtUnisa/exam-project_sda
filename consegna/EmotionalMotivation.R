#==============================   CONFIG     ============================

ABS_PATH = 'C:/Users/gorra/Desktop/gitSDA_Def/SDAgruppo2'
DATASET_FILENAME = 'RegressionData_final.csv'
Y_LABEL = 'Y_EmotionalMotivation'
PREDICTORS_NUMBER = 10

#================================ START =================================

setwd(ABS_PATH)
source('./utils.R')
ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)


#==================== REGRESSION WITHOUT INTERACTIONS ====================

baseModel=lm.byIndices(ds, -1)
lm.inspect(baseModel, 10)
#R-squared:  0.9845  MSE=3.508746

#====================== INSPECT RELATIONSHIPS ============================

showPlotsAgainstOutput(ds, 2:(PREDICTORS_NUMBER+1))

#======================== TEST RELATIONSHIPS =============================

possibleDependencies = list('I(X_AvgPlayerValue^2)',
                            'X_RestTimeFromLastMatch',
                            'X_MatchRelevance',
                            'X_AvgGoalConcededLastMatches',
                            'X_SupportersImpact',
                            'X_OpposingSupportersImpact')

dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(dependencyModel, 5) 
#MSE = 29.97714  R-squared:   0.8481

#======================== INSPECT INTERACTIONS =============================

# Collect rsquared for every linear model obtained by adding every possible
# interaction between two distinct predictors to the base model.
# Set base rsquared as default value
#
# The matrix will be store as an upper triangular matrix 
# for computational efficiency

baseRSquared = summary( lm.byIndices(ds, -1) )$r.squared
interactionMatrix = inspectInteractionMatrix(ds, default=baseRSquared, showHeatmap = T)


#====================  BEST SUBSET SELECTION WITH INTERACTIONS   ===============

# add non linearities for best subset selection
possibleRelationships = list('X_MatchRelevance*X_OpposingSupportersImpact',
                             'X_MatchRelevance*X_Altitude',
                             'I(X_AvgPlayerValue^2)',
                             'X_SupportersImpact*X_AvgGoalConcededLastMatches'
)    

bestSubsets = bestSubsetSelection(ds, relationships=possibleRelationships, nMSE=10, folds=10, verbose=T, method="exhaustive")
ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

bestSubset = bestSubsets$model[[6]]
bestSubsetMSE = bestSubsets$MSE[[6]]
lm.inspect(bestSubset, 10, 10)

# [1] "================= SUMMARY ================="
# 
# Call:
#   lm(formula = f, data = data, x = TRUE, y = TRUE)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -3.5709 -1.1787  0.0515  1.0668  4.1300 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)                   -0.2692     0.1742  -1.546    0.126    
# X_RestTimeFromLastMatch        1.2330     0.1833   6.728 1.38e-09 ***
#   X_AvgPlayerValue               4.9987     0.1765  28.324  < 2e-16 ***
#   X_MatchRelevance               9.1352     0.1644  55.554  < 2e-16 ***
#   X_AvgGoalConcededLastMatches   1.8667     0.1806  10.338  < 2e-16 ***
#   X_SupportersImpact             5.1158     0.1664  30.748  < 2e-16 ***
#   X_OpposingSupportersImpact    -1.9385     0.1791 -10.822  < 2e-16 ***
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Residual standard error: 1.706 on 93 degrees of freedom
# Multiple R-squared:  0.9842,	Adjusted R-squared:  0.9832 
# F-statistic: 965.8 on 6 and 93 DF,  p-value: < 2.2e-16
# 
# [1] "==================  MSE  =================="
# [1] 3.125911


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
# -3.4953 -1.0502  0.0857  0.9081  4.2295 
# 
# Coefficients:
# Estimate Std. Error t value Pr(>|t|)    
# (Intercept)                                      -0.2234     0.1700  -1.315   0.1919    
# X_RestTimeFromLastMatch                           1.2251     0.1779   6.886 6.91e-10 ***
# X_AvgPlayerValue                                  4.9417     0.1727  28.612  < 2e-16 ***
# X_MatchRelevance                                  9.1611     0.1599  57.284  < 2e-16 ***
# X_OpposingSupportersImpact                       -1.9601     0.1741 -11.261  < 2e-16 ***
# X_SupportersImpact                                5.1727     0.1630  31.737  < 2e-16 ***
# X_AvgGoalConcededLastMatches                      1.9589     0.1788  10.953  < 2e-16 ***
# X_SupportersImpact:X_AvgGoalConcededLastMatches   0.4153     0.1604   2.590   0.0112 *  
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Residual standard error: 1.656 on 92 degrees of freedom
# Multiple R-squared:  0.9853,	Adjusted R-squared:  0.9842 
# F-statistic: 879.6 on 7 and 92 DF,  p-value: < 2.2e-16
# 
# [1] "==================  MSE  =================="
# [1] 2.90048
# QUESTO é QUELLO MIGLIORE PER ORA

#============================  RIDGE E LASSO   =================================

# Add non linearities before scaling
bestInteractions = list('X_MatchRelevance*X_OpposingSupportersImpact',
                        'X_MatchRelevance*X_Altitude',
                        'I(X_AvgPlayerValue^2)',
                        'X_SupportersImpact*X_AvgGoalConcededLastMatches'
)     

ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)
ds_scaled = addNonLinearities(ds, bestInteractions)

lambda_grid = 10^seq(4, -6, length = 2000)

models = lm.shrinkage(ds_scaled, lambda_grid, nMSE=2, folds=4, showPlot=T)
coef(models$lasso$model, s = models$lasso$bestlambda)
coef(models$ridge$model, s = models$ridge$bestlambda)

min(models$ridge$cvm)
# MSE=3.124632

min(models$lasso$cvm)
# MSE=3.10594

# LASSO E RIDGE VANNO MALE RISPETTO AL PRECEDENTE


# predictedY = predictWithGlmnet(models$lasso, newx=as.matrix(ds_scaled[,-1]))


#============================= ELASTIC NET  ===============================

bestInteractions = list('X_MatchRelevance*X_OpposingSupportersImpact',
                        'X_MatchRelevance*X_Altitude',
                        'I(X_AvgPlayerValue^2)',
                        'X_SupportersImpact*X_AvgGoalConcededLastMatches'
)     

ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)
ds_scaled = addNonLinearities(ds, bestInteractions)

lambda_grid = 10^seq(4, -6, length = 500)
alpha_grid = seq(0,1,length = 100)


MSEs = lm.elasticNet(ds_scaled, alpha_grid, lambda_grid, nMSE=10, folds=10, best_mse = bestSubsetMSE, showPlot = T, verbose = T)
min(MSEs)   #MSE = 3.000327
# ANCHE ELASTIC NET VA PEGGIO DEL MODELLO TROVATO IN PRECEDENZA 

# lm.plotElasticNet(alpha_grid, MSEs, 2)

#======================= LINEAR REGRESSION - ISSUES =======================
# the best model to analyze

best_model = lm(Y_EmotionalMotivation ~ X_AvgPlayerValue + X_RestTimeFromLastMatch +
                X_MatchRelevance + X_OpposingSupportersImpact + X_SupportersImpact +
                X_AvgGoalConcededLastMatches + X_SupportersImpact*X_AvgGoalConcededLastMatches, data=ds, y=T,x=T)

lm.inspect(best_model, 10)

# 1) non-linearities & homoschedasticity ----------------------------------
# analyze residuals
plot(best_model, which=1)

# 2) high leverage points -------------------------------------------------

hats = hat.plot(best_model)
hats_indices = c(12, 44)


# 4) collinearity ---------------------------------------------------------

# check any collinearity
collinearity_models = vifs.plot(ds)

# 4) outliers -------------------------------------------------------------

outlier_indices = outlier.plot(best_model)

if(length(hats_indices) > 0) {
  ds_without_leverage = ds[-hats_indices,]
}

refitted_best_model = lm.refit(best_model, ds_without_leverage)

lm.inspect(refitted_best_model, 10, 10)

#plotres(models$lasso$model, s=models$lasso$bestlambda)



#======================= CONCLUSION =======================

best_formula = "Y_EmotionalMotivation ~ X_AvgPlayerValue + X_RestTimeFromLastMatch +
                X_MatchRelevance + X_OpposingSupportersImpact + X_SupportersImpact +
                X_AvgGoalConcededLastMatches + X_SupportersImpact*X_AvgGoalConcededLastMatches"

best_summary = '
            [1] "================= SUMMARY ================="
            
            Call:
            lm(formula = formula(model), data = data, x = T, y = T)
            
            Residuals:
                Min      1Q  Median      3Q     Max 
            -3.5299 -1.0701  0.0594  0.9176  4.1845 
            
            Coefficients:
                                                            Estimate Std. Error t value Pr(>|t|)    
            (Intercept)                                      -0.2159     0.1728  -1.250   0.2146    
            X_AvgPlayerValue                                  4.9490     0.1748  28.310  < 2e-16 ***
            X_RestTimeFromLastMatch                           1.2221     0.1818   6.722 1.59e-09 ***
            X_MatchRelevance                                  9.1570     0.1653  55.385  < 2e-16 ***
            X_OpposingSupportersImpact                       -1.9924     0.1813 -10.989  < 2e-16 ***
            X_SupportersImpact                                5.1344     0.1747  29.396  < 2e-16 ***
            X_AvgGoalConcededLastMatches                      1.9748     0.1822  10.840  < 2e-16 ***
            X_SupportersImpact:X_AvgGoalConcededLastMatches   0.3913     0.1754   2.231   0.0282 *  
            ---
            Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
            
            Residual standard error: 1.67 on 90 degrees of freedom
            Multiple R-squared:  0.985,	Adjusted R-squared:  0.9839 
            F-statistic: 845.4 on 7 and 90 DF,  p-value: < 2.2e-16
            
            [1] "==================  MSE  =================="
            [1] 2.98143
'
# OTTENUTO CON BEST SUBSET

best_model = lm(best_formula, data=ds_without_leverage,y=T,x=T)
lm.inspect(best_model, 10, 10)

exportCOEF(best_model$coefficients)
