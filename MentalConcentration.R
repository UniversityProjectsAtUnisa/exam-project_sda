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

#============================== functions ===============================

#compute MSE , media su più cross validation

mean_cvMSE = function(model, n=1000, k=5){
  MSEs = vector(mode='numeric',n)
  for (i in 1:n){
    MSEs[i] = cv.lm(model, k = k)$MSE$mean
  }
  return(mean(MSEs))
}

lmByIndices <- function (dataset, indices, addIntercept=T) {
  f <- paste(names(dataset)[Y_index], "~")
  if(!addIntercept){
    f <- paste(f,"0 + ")
  }
  f <- paste(f, paste(names(dataset)[indices], collapse=" + "))
  return(lm(f, data=myds, y=T, x=T))
}

mse = function(predicted,actual){ mean( ( predicted-actual )^2 ) }

mostra_grafici = function(dataset, predictors, y_index){
  for (i in 1:predictors){
    plot(dataset[,i], dataset[,y_index], ylab=names(dataset)[y_index],xlab=names(dataset)[i])
    readline(prompt="Press [enter] to continue")
  }
}

#============================== start     ===============================

setwd('C:\\Users\\carbo\\OneDrive\\Documenti\\Magistrale Carbone\\2 sem\\Statistical Data Analysis\\aPROGETTO')
myds=read.csv('RegressionData_SDA_AH_group2.csv') 
myds=na.omit(myds)
col_names <- colnames(myds)

# Inserire interazioni nella matrice prima di scalare
#interazioni
i_vector = c('X_Temperature'   , 'X_SupportersImpact', 'X_AvgGoalConcededLastMatches', 'X_MatchRelevance' )
j_vector = c('X_AvgPlayerValue', 'X_AvgPlayerValue'  , 'X_AvgPlayerValue',             'X_MatchRelevance')

myds_col_names = names(myds)

for(k in 1:length(i_vector)){
  i = which(names(myds)== i_vector[k]) 
  j = which(names(myds)== j_vector[k]) 
  myds = cbind(myds, myds[,i]*myds[,j])
  myds_col_names = c(myds_col_names, paste(myds_col_names[i], "*", myds_col_names[j],sep = ""))
}
names(myds) = myds_col_names

###################
Y_index = 17
myds = cbind(myds[,1:10], myds[,19:length(myds)],myds[,Y_index])
names(myds)[length(myds)]='Y_MentalConcentration'
myds_scaled=data.frame()

myds_scaled=cbind(as.data.frame(scale(myds[,1:14])), myds[,15])

col_names <- colnames(myds)
col_names_scaled = colnames(myds_scaled)
colnames(myds_scaled) = col_names
Y_index = 17
predictors = 10

#========= regressione senza interazioni =================

reg_base=lmByIndices(myds, 1:predictors)
rsquared=summary(reg_base)$r.squared

# analizzo le relazioni
mostra_grafici(myds, predictors, Y_index)
#dall'analisi dei grafici vedo che
#X_Temperature -> quadrato
#X_AvgPlayerValue -> forma a S, log?
#X_MatchRelevance -> quadrato?
#supportersImpact -> sqrt?

model_complesso = lm(Y_MentalConcentration ~
                      #I(X_Temperature^2)
                    +  X_RestTimeFromLastMatch
                     + (X_AvgPlayerValue)
                     + I(X_MatchRelevance^2)
                     #+ sqrt(X_SupportersImpact)
                     ,data=myds, y=TRUE,x=TRUE)
summary(model_complesso)

plot(model_complesso, which=1)


mean_cvMSE(model_complesso, 5)
mean_cvMSE(reg_base, 5)

#model complesso va meglio


############## interazioni ##################
# visualizzo le interazioni rilevanti

interaction_matrix=matrix(0, predictors-1, predictors-1) # matrice di R^2 con tutte le interazioni
for(i in 1:(predictors-1) ){
  for(j in (i+1):predictors){
    f <- paste(names(myds)[Y_index], "~", paste(names(myds)[-( (predictors+1):ncol(myds) )], collapse=" + "))# "v1 ~ v2 + v3 + v4"
    f=paste(f,"+",names(myds)[i],"*",names(myds)[j])
    mylm=lm(f, data=myds)
    interaction_matrix[i,j-1]=summary(mylm)$r.squared
  }
}

colnames(interaction_matrix) <- col_names[2:predictors]
rownames(interaction_matrix) <- col_names[1:(predictors-1)]

View(interaction_matrix)

pheatmap(as.data.frame(interaction_matrix),
         display_numbers = T, 
         #breaks = seq(0.8,0.9,length=10),
         color = colorRampPalette(c("navy", "white", "firebrick3"))(50))

#prova interazione: modello complesso + X_Temperature * X_AvgPlayerValue
model_interazioni_temp_value = lm(Y_MentalConcentration ~
                       +  X_RestTimeFromLastMatch
                       + (X_AvgPlayerValue)
                       + I(X_MatchRelevance^2)
                       + X_Temperature * X_AvgPlayerValue
                     ,data=myds, y=TRUE,x=TRUE)
summary(model_interazioni_temp_value)

mean_cvMSE(model_interazioni_temp_value, 10,10) # MSE=1.779052
reg_base_mse = mean_cvMSE(reg_base, 5)

#prova interazione: modello complesso + X_AvgGoalConcededLastMatches * X_AvgPlayerValue

model_interazioni_goal_value = lm(Y_MentalConcentration ~
                                    +  X_RestTimeFromLastMatch
                                  + (X_AvgPlayerValue)
                                  + I(X_MatchRelevance^2)
                                  + X_AvgGoalConcededLastMatches * X_AvgPlayerValue
                                  ,data=myds, y=TRUE,x=TRUE)
summary(model_interazioni_goal_value)

mean_cvMSE(model_interazioni_goal_value, 5) #MSE=2.042071

#prova interazione: modello complesso + X_AvgGoalConcededLastMatches * X_AvgPlayerValue
#                                     + X_Temperature * X_AvgPlayerValue

model_interazioni_goal_temp_value = lm(Y_MentalConcentration ~
                                    +  X_RestTimeFromLastMatch
                                  + (X_AvgPlayerValue)
                                  + I(X_MatchRelevance^2)
                                  + X_AvgGoalConcededLastMatches * X_AvgPlayerValue
                                  + X_Temperature * X_AvgPlayerValue
                                  ,data=myds, y=TRUE,x=TRUE)
summary(model_interazioni_goal_temp_value)

mean_cvMSE(model_interazioni_goal_temp_value, 5) #MSE=1.829902

#prova interazione: modello compelsso + X_SupportersImpact * X_AvgPlayerValue
#+ X_AvgGoalConcededLastMatches * X_AvgPlayerValue
#+ X_Temperature * X_AvgPlayerValue
# varie combinazioni
model_interazioni_supp_value= lm(Y_MentalConcentration ~
                                         +  X_RestTimeFromLastMatch
                                       + (X_AvgPlayerValue)
                                       + I(X_MatchRelevance^2)
                                       + X_AvgGoalConcededLastMatches * X_AvgPlayerValue
                                       + X_SupportersImpact * X_AvgPlayerValue
                                       #+ X_Temperature * X_AvgPlayerValue
                                       ,data=myds, y=TRUE,x=TRUE)
summary(model_interazioni_supp_value)

mean_cvMSE(model_interazioni_supp_value, 10, 10) 
#MSE no temp_value      : 1.492497 - 1.439522
#MSE no goal_value      : 1.660607
#MSE no supporter_value : 1.921177
#MSE tutte e 3 le interazioni : 1.706468

#il miglior modello è model_interazioni_supp_value



# ========== best subset con interazioni =================
predictors=10
Y_index = 17
myds_col_names = names(myds)[1:predictors]
myds_interactions = data.frame(myds[,1:predictors])

#imposto le interazioni con due vettori (interazione i_vector[k]*j_vector[k])
i_vector = c('X_Temperature'   , 'X_SupportersImpact', 'X_AvgGoalConcededLastMatches', 'X_MatchRelevance')#,'X_OpposingSupportersImpact' )
j_vector = c('X_AvgPlayerValue', 'X_AvgPlayerValue'  , 'X_AvgPlayerValue',             'X_MatchRelevance')#,'X_AvgPlayerValue')

for(k in 1:length(i_vector) ){
    i = which(names(myds)== i_vector[k]) 
    j = which(names(myds)== j_vector[k]) 
    myds_interactions = cbind(myds_interactions, myds[,i]*myds[,j])
    if (myds_col_names[i]==myds_col_names[j]){
      myds_col_names = c(myds_col_names, paste("I(",myds_col_names[i],"^2)"))
    }else{
    myds_col_names = c(myds_col_names, paste(myds_col_names[i],myds_col_names[j], sep = "*"))
    }
    
}
names(myds_interactions) = myds_col_names
myds_interactions = cbind(myds_interactions, (myds[,Y_index])) 
names(myds_interactions) = c(myds_col_names, names(myds)[Y_index])
View(myds_interactions)

#

combination_number = (2^(length(myds_interactions)-1))-1
subsets = matrix(list(), 1, combination_number)
Y_index=length(myds_interactions)
predictors=length(myds_interactions)-1
for(comb in 1:combination_number){
  f=paste(names(myds_interactions)[Y_index], "~")
  for(k in 1:predictors){
    if( bitwAnd( comb, 2^(k-1) ) > 0 ) {
      f=paste(f, names(myds_interactions)[k], "+")
    }
  }
  f = str_sub(f,1,nchar(f)-1) # Rimuovo l'ultimo +
  #print(f)
  subsets[1,comb]=list(lm(f, data=myds, y=TRUE, x=TRUE))
}

# Migliori subsets per numero di regressori
best_subsets=vector(mode="list",length=predictors)

for(elem in subsets[1,]) {
  index = length(elem$coefficients)-1
  rsquared = summary(elem)$r.squared
  if(is.null(best_subsets[[index]])){
    best_subsets[[index]] = elem
  }
  else if(rsquared > summary(best_subsets[[index]])$r.squared){ # R non esegue lo short circuit
    best_subsets[[index]] = elem
  }
}



best_MSEs = vector(length=predictors)
temp_lm = 0
# Sceglie il migliore tra tutti i modelli senza interazioni
# (Cross-Validation)
for(i in 1:length(best_subsets)) {
  print(i)
  cnames = colnames(best_subsets[[i]]$model)
  yname = cnames[1]
  xnames = colnames(summary(best_subsets[[i]])$cov.unscaled)[-1]
  xnames = map(xnames, function(x){str_replace(x,":",'*')})
  f = paste(xnames, collapse = ' + ')
  f = paste(yname, "~", f)
  temp_lm = lm(f, data=myds_interactions, y=T, x=T)
  print(f)
  best_MSEs[i] = mean_cvMSE(temp_lm, 5)
}

best_model = best_subsets[[which.min(best_MSEs)]]

plot(best_MSEs) #il best model è proprio quello che avevo trovato prima con i 
                #vari tentativi sulle interazioni
mean_cvMSE(best_model, 10, 10)


############# RIDGE E LASSO - ELASTIC NET #########################

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


