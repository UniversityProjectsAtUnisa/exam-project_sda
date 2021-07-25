library(matrixcalc)
library(car)
library(plotly)
library(stringr)
library(glmnet)
library(lmvar)
library(glmnet)
library(MASS)
library(parallel)
library(purrr)
library(olsrr)
library(ggcorrplot)
library(zeallot)
library(pheatmap)
library(leaps)
library(MuMIn)
source('./utils.R')

#==============================   CONFIG     ============================

ABS_PATH = 'C:/Users/marco/Documents/UNISA/SDA/progetto/SDAgruppo2'
DATASET_FILENAME = 'RegressionData_SDA_AH_group2.csv'
Y_LABEL = 'Y_MentalConcentration'
PREDICTORS_NUMBER = 10

#================================ START =================================

setwd(ABS_PATH)
ds = ds.init(DATASET_FILENAME, Y_LABEL, PREDICTORS_NUMBER)


#==================== REGRESSION WITHOUT INTERACTIONS ====================

baseModel=lm.byIndices(ds, -1)
lm.inspect(baseModel, 5)


#====================== INSPECT RELATIONSHIPS ============================

showPlotsAgainstOutput(ds, 2:PREDICTORS_NUMBER+1)



#======================== TEST RELATIONSHIPS =============================

possibleDependencies = list('X_RestTimeFromLastMatch', 
                            'X_AvgPlayerValue', 
                            'I(X_MatchRelevance^2)')

dependencyModel = lm.byFormulaChunks(ds, possibleDependencies)
lm.inspect(modelWithPossibleDependencies, 5)



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

possibleDependencies = list('X_RestTimeFromLastMatch', 'X_AvgPlayerValue', 'I(X_MatchRelevance^2)')

possibleInteractions = list('X_Temperature*X_AvgPlayerValue')
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(dependencyModelWithPossibleInteractions, 5)


possibleInteractions = list('X_AvgGoalConcededLastMatches*X_AvgPlayerValue')
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(dependencyModelWithPossibleInteractions, 5)


possibleInteractions = list('X_AvgGoalConcededLastMatches*X_AvgPlayerValue', 'X_Temperature*X_AvgPlayerValue')
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(dependencyModelWithPossibleInteractions, 5)


possibleInteractions = list(
  'X_SupportersImpact*X_AvgPlayerValue', 
  'X_AvgGoalConcededLastMatches*X_AvgPlayerValue', 
  'X_Temperature*X_AvgPlayerValue'
)
dependencyModelWithPossibleInteractions = lm.byFormulaChunks(ds, append(possibleDependencies, possibleInteractions))
lm.inspect(modelWithPossibleDependencies, 5)


#====================  BEST SUBSET SELECTION WITH INTERACTIONS   ===============

# add non linearities for best subset selection
possibleInteractions = list(
  'X_Temperature*X_AvgPlayerValue',
  'X_AvgPlayerValue*X_SupportersImpact',
  'X_AvgGoalConcededLastMatches*X_AvgPlayerValue'
)    

bestSubsets = bestSubsetSelection(ds, relationships=possibleRelationships, nMSE=2, folds=2, verbose=T, method="exhaustive")
ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

bestSubset = bestSubsets$model[[which.min(bestSubsets$MSE)]]
bestSubsetOSE = oneStandardErrorSubset(bestSubsets)

plot(bestSubsetOSE, 1)


#===============  RIDGE E LASSO - ELASTIC NET   ===================

############# ADD NON LINEARITIES BEFORE SCALING ##################

bestInteractions = list(
  'X_Temperature*X_AvgPlayerValue',
  'X_AvgPlayerValue*X_SupportersImpact',
  'X_AvgGoalConcededLastMatches*X_AvgPlayerValue'
)    
ds = ds.scale(addNonLinearities(ds, bestInteractions))


lambda_grid = 10^seq(10, -3, length = 2000)
#set.seed(1)

x = as.matrix(myds_scaled[,1:(length(myds_scaled)-1)])

shrinkage_MSEs = matrix(0,2,length(lambda_grid))
n_iterations=10
Y_index=15
for (i in 1:n_iterations){
  
  ridge = cv.glmnet(x, as.matrix(myds[,Y_index]),
                    alpha=0, lambda = lambda_grid, nfolds=5, trace.it = 0)
  lasso = cv.glmnet(x, as.matrix(myds[,Y_index]),
                    alpha=1, lambda = lambda_grid, nfolds=5, trace.it = 0)
  
  shrinkage_MSEs[1,] =shrinkage_MSEs[1,]+ ridge$cvm
  shrinkage_MSEs[2,] =shrinkage_MSEs[2,]+ lasso$cvm
}
coef(ridge, s = ridge$lambda.min)
coef(lasso, s = lasso$lambda.min)

print('migliori lambda ridge')
print(lambda_grid[which.min(shrinkage_MSEs[1,])])
print('migliori lambda lasso')
print(lambda_grid[which.min(shrinkage_MSEs[2,])])


bestmse = mean_cvMSE(best_model,5)# cv.lm(best_model,k=5)$MSE$mean

plot(lambda_grid,shrinkage_MSEs[2,]/n_iterations,col = 'orange',log='x',type='l')
lines(lambda_grid,shrinkage_MSEs[1,]/n_iterations,xlab=expression(lambda),ylab = 'cvMSE',pch = 21,type='l',col = 'blue')


legend('bottomright', legend=c('Ridge', 'Lasso'), col=c('blue','orange'), pch=20)


best_index_lasso = which.min(shrinkage_MSEs[2,])

lasso = cv.glmnet(x, as.matrix(myds[,Y_index]),
                  alpha=1, lambda = c( lambda_grid[best_index_lasso],10000000), trace.it = 0, nfolds = 5)

coef(lasso, s = lasso$lambda.min)



min(shrinkage_MSEs[1,]/n_iterations)
min(shrinkage_MSEs[2,]/n_iterations)


#elastic net

alpha_grid = seq(0,1,length = 1000)
MSEs = vector(mode='numeric', 1000)
lambdas = vector(mode='numeric', 1000)


i = 1
foldids = sample(rep(seq(5), length = 40))
for(alpha in alpha_grid){
  model = cv.glmnet(x, as.matrix(myds_scaled[,Y_index]), alpha=alpha, lambda=lambda_grid, foldid=foldids)
  MSEs[i] = min(model$cvm)
  i = i + 1
  print(i)
}


plot(alpha_grid,MSEs,type='l')
lines(c(min(lambda_grid), max(lambda_grid)), c(bestmse,bestmse))

which.min(mses)
alpha_grid[which.min(mses)]

############## PROBLEMI DELLA REG LINEARE ############
# 1) NON LINEARITÀ
# vediamo i residui
plot(best_model, which=1)
plot(best_model)
# La linea rossa non è dritta quindi c'è della non linearità che non è stata spiegata

# 2) Outliers
stud_res=studres(best_model)
plot(best_model$fitted.values, stud_res)

# ci sono outlier : osservazione n° 35
myds = myds[-35,]
#re-fit
best_model= lm(Y_MentalConcentration ~
                                   +  X_RestTimeFromLastMatch
                                 + (X_AvgPlayerValue)
                                 + I(X_MatchRelevance^2)
                                 + X_AvgGoalConcededLastMatches * X_AvgPlayerValue
                                 + X_SupportersImpact * X_AvgPlayerValue
                                 #+ X_Temperature * X_AvgPlayerValue
                                 ,data=myds, y=TRUE,x=TRUE)
summary(best_model)
mean_cvMSE(best_model, 10, 10)
w# 3) high leverage points ------ 
hats <- as.data.frame(hatvalues(best_model))
# non c'è nessun valore >>(p+1)/n


