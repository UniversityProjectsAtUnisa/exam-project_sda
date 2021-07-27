library(matrixcalc)
library(car)
library(plotly)
library(stringr)
library(glmnet)
library(lmvar)
library(MASS)
library(parallel)
library(purrr)
library(olsrr)
library(ggcorrplot)
library(zeallot)
library(pheatmap)
library(leaps)
library(MuMIn)
library(scales)
library(VGAM)
library(matlib)
library(ggfortify)
source('./utils.R')

#==============================   CONFIG     ============================

ABS_PATH = 'C:/Users/marco/Documents/UNISA/SDA/progetto/SDAgruppo2'
DATASET_FILENAME = 'RegressionData_SDA_AH_group2.csv'
Y_LABEL = 'Y_MentalConcentration'
PREDICTORS_NUMBER = 10

OUTLIER_STUDENTIZED_RES_THRESHOLD = 3

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
possibleRelationships = list(
  'X_Temperature*X_AvgPlayerValue',
  'X_AvgPlayerValue*X_SupportersImpact',
  'X_AvgGoalConcededLastMatches*X_AvgPlayerValue'
)    

bestSubsets = bestSubsetSelection(ds, relationships=possibleRelationships, nMSE=10, folds=10, verbose=T, method="exhaustive")
ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")

bestSubset = bestSubsets$model[[which.min(bestSubsets$MSE)]]
bestSubsetOSE = oneStandardErrorSubset(bestSubsets)

plot(bestSubsetOSE, 1)



#============  BEST SUBSET SELECTION WITH INTERACTIONS FORWARD   ===============


possibleRelationships = list(
  'I(X_MatchRelevance^2)'
)
bestSubsets = bestSubsetSelection(ds, relationships=possibleRelationships, nMSE=2, folds=2, method="forward", nvmax=15, verbose=T)

ds.prettyPlot(bestSubsets$MSE, xlab="Number of predictors", ylab="CV test MSE", title="5-fold cross-validation Test MSE")


#===============  RIDGE E LASSO - ELASTIC NET   ===================

############# ADD NON LINEARITIES BEFORE SCALING ##################

bestInteractions = list(
  'X_Temperature*X_AvgPlayerValue',
  'X_AvgPlayerValue*X_SupportersImpact',
  'X_AvgGoalConcededLastMatches*X_AvgPlayerValue'
)    
ds_scaled = ds.scale(addNonLinearities(ds, bestInteractions))
lambda_grid = 10^seq(4, -6, length = 2000)

models = lm.shrinkage(ds_scaled, lambda_grid, nMSE=2, folds=4, showPlot=T)

coef(lasso, s = lasso$lambda.min)


#============================= ELASTIC NET  ===============================

bestInteractions = list(
  'X_Temperature*X_AvgPlayerValue',
  'X_AvgPlayerValue*X_SupportersImpact',
  'X_AvgGoalConcededLastMatches*X_AvgPlayerValue'
)    
ds_scaled = ds.scale(addNonLinearities(ds, bestInteractions))

lambda_grid = 10^seq(4, -6, length = 2000)
alpha_grid = seq(0,1,length = 100)

MSEs = lm.elasticNet(ds_scaled, alpha_grid, lambda_grid, nMSE=300, folds=10, best_mse = 2, showPlot = T, verbose = T)

# lm.plotElasticNet(alpha_grid, MSEs, 2)

#======================= LINEAR REGRESSION - ISSUES =======================
# the best model to analyze
best_model = bestSubset # or any other

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

hat.plot <- function(fit, interactive=T) {
  p <- length(coefficients(fit))
  n <- length(fitted(fit))
  hats=hatvalues(fit)
  boundaries = c(2,3)*p/n
  ggplotDF =  cbind(c(1:length(hats)),
                    data.frame(hats), 
                    rep(boundaries[1], length(hats)),
                    rep(boundaries[2], length(hats)))
  names(ggplotDF) = c("point", "Hat",'LowerBound', 'UpperBound')
  ggplot(ggplotDF,aes(point, Hat))+
    geom_point(shape=21, color="#86bbd8", fill="#86bbd8", size=2.5)+
    geom_line(aes(point, LowerBound), color="red")+
    geom_line(aes(point, UpperBound), color="red")+
    labs(x = 'Points',
         y = 'Hat values')
  
  if (interactive) ggplotly()
}
hat.plot(best_model)

# 4) collinearity ---------------------------------------------------------
vifs.plot <- function(ds, predictors_indices){
  #da modificare e tenere in considerazione anche le intearzioni. ci devo pensare
  
  ggplotDF = (data.frame(Predictor=integer(), VIF=integer()))
  
  for (index in predictors_indices){
    predictor_to_fit=names(ds)[index]
    formula = paste(predictor_to_fit, '~')
    pred_indices = predictors_indices[predictors_indices!=index]
    predictors = paste(names(ds)[pred_indices],collapse='+')
    formula = paste(formula, predictors)
    #print(formula)
    model = lm(formula, data=ds)
    r.squared = summary(model)$r.squared
    vif = 1/(1-r.squared)
    
    
    ggplotDF=rbind(ggplotDF, c(predictor_to_fit, vif))
  }
  
  names(ggplotDF) = c('Predictor','VIF')
  print(ggplotDF)
  gg <- ggplot(data=ggplotDF, aes(x=Predictor,y=VIF))+geom_bar(stat="identity")+coord_flip()+
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
  
  print(gg)
}

#check any collinearity
vifs.plot(ds, 2:length(ds))

# 4) outliers -------------------------------------------------------------

outlier.plot <- function(fit, interactive=T) {
  stud_res=studres(best_model) #studentized residuals
  boundaries = c(-3,3)
  ggplotDF =  cbind(c(1:length(stud_res)),
                    data.frame(stud_res), 
                    rep(boundaries[1], length(hats)),
                    rep(boundaries[2], length(hats)))
  names(ggplotDF) = c("point", "studentized_residual",'LowerBound', 'UpperBound')
  ggplot(ggplotDF,aes(point, studentized_residual))+
    geom_point(shape=21, color="#86bbd8", fill="#86bbd8", size=2.5)+
    geom_line(aes(point, LowerBound), color="red")+
    geom_line(aes(point, UpperBound), color="red")+
    labs(x = 'Points',
         y = 'studentized residual')
  
  if (interactive) ggplotly()
  
  outliers_indices = as.vector(which(abs(stud_res) > OUTLIER_STUDENTIZED_RES_THRESHOLD))
  return(outliers_indices)
}
outlier.plot(best_model)
#outliers_indices = which(abs(stud_res) > OUTLIER_STUDENTIZED_RES_THRESHOLD)

# remove outliers from dataset and re-fit the model
# ...













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


