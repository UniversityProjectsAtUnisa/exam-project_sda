#================ Config constants =====================

POLYNOMIAL_REGEX  = '.*I\\((.*)\\^(\\d+)\\)'
FUNCTIONAL_REGEX  = '(\\w*)\\((\\w*)\\)'
INTERACTION_REGEX = '(\\w+)\\*(\\w+)'

#================ Config variables =====================

utils.PREDICTORS_NUMBER = NULL
utils.Y_LABEL = NULL

#=======================================================

#compute MSE , media su più cross validation

mean_cvMSE = function(model, n=1000, k=5){
  MSEs = vector(mode='numeric',n)
  for (i in 1:n){
    MSEs[i] = cv.lm(model, k = k)$MSE$mean
  }
  return(mean(MSEs))
}

lm.byIndices <- function (data, indices, addIntercept=T) {
  f <- paste(utils.Y_LABEL, "~")
  if(!addIntercept){
    f <- paste(f,"0 + ")
  }
  f <- paste(f, paste(names(data)[indices], collapse=" + "))
  print(f)
  return(lm(f, data=data, y=T, x=T))
}

lm.byFormulaChunks <- function(data, chunks) {
  lm(paste(utils.Y_LABEL, " ~ ", paste(possibleDependencies, collapse = ' + ')), data=ds, y=T, x=T)
}


mse = function(predicted,actual){ mean( ( predicted-actual )^2 ) }

showPlotsAgainstOutput = function(data, indices){
  for (i in indices){
    plot(data[,i], data[,utils.Y_LABEL], ylab=utils.Y_LABEL,xlab=names(data)[i])
    readline(prompt="Press [enter] to continue")
  }
}

ds.init = function (filename, y_label, predictors_number) {
  utils.Y_LABEL <<- y_label
  utils.PREDICTORS_NUMBER <<- predictors_number
  
  return(requireDataset(filename))
}

requireDataset = function (filename) {
  ds = read.csv(filename)
  ds = na.omit(ds)
  ds = cbind(ds[utils.Y_LABEL], ds[1:utils.PREDICTORS_NUMBER])
  return(ds)
}

ds.scale = function(ds) {
  # Assumes the only dependent variable is 
  # in the first column
  return(cbind(ds[1], scale(ds[2:length(ds)])))
}

lm.inspect = function(model, nMSE=1000, folds=5) {
  z <- list()
  
  print("================= SUMMARY =================")
  z$summary <- summary(model)
  print(z$summary)
  
  print("==================  MSE  ==================")
  z$MSE <- mean_cvMSE(model, nMSE, folds)
  print(z$MSE)
        
  plot(model, which=1)
  
  invisible(z)
}

addInteractions = function(data, chunks) {
  interaction_matrix = NULL
  splitter = '\\*'
  
  for(chunk in chunks) {
    elements = str_split(chunk, splitter, simplify=T)
    first_name = str_trim(elements[[1]])
    second_name = str_trim(elements[[2]])
    
    if(is.null(interaction_matrix)){
      interaction_matrix = data[,first_name]*data[,second_name]
    } else {
      interaction_matrix = cbind(interaction_matrix, data[,first_name]*data[,second_name])
    }
  }
  
  return(cbind(data, interaction_matrix))
}



addPolynomials = function(data, chunks) {
  polynomial_matrix = NULL
  r = POLYNOMIAL_REGEX
  
  
  for(chunk in chunks) {
    elements = str_match(chunk, r)[1,]
    if(any(is.na(elements))){
      stop(paste("chunks do not match regex: ", chunk))
    }
    predictor_name = elements[2]
    power = strtoi(elements[3], 10)
    if(power == 0) {
      stop('power is not valid')
    }
    
    new_column = data[, predictor_name] ^ power
    
    if(is.null(polynomial_matrix)){
      polynomial_matrix = new_column
    } else {
      polynomial_matrix = cbind(polynomial_matrix, new_column)
    }
  }
  
  return(cbind(data, polynomial_matrix))
}

addFunctions = function(data, chunks) {
  functional_matrix = NULL
  r = FUNCTIONAL_REGEX
  
  
  for(chunk in chunks) {
    elements = str_match(chunk, r)[1,]
    if(any(is.na(elements))){
      stop(paste("chunks do not match regex: ", chunk))
    }
    function_name = elements[2]
    predictor_name = strtoi(elements[3], 10)
    
    func = get(function_name)
    new_column = func(data[, predictor_name])
    
    if(is.null(functional_matrix)){
      functional_matrix = new_column
    } else {
      functional_matrix = cbind(functional_matrix, new_column)
    }
  }
  
  return(cbind(data, functional_matrix))
}

addNonLinearities = function(data, chunks) {
  polynomial_regex = POLYNOMIAL_REGEX
  functional_regex = FUNCTIONAL_REGEX
  
  col_names = names(data)
  polynomial_chunks = str_match(chunks, polynomial_regex)[, 1]
  polynomial_chunks = polynomial_chunks[!is.na(polynomial_chunks)]
  if(length(polynomial_chunks) > 0) {
    data = addPolynomials(data, polynomial_chunks)
    names(data) = c(col_names, polynomial_chunks)
    chunks = setdiff(chunks, polynomial_chunks)
  }
  
  col_names = names(data)
  functional_chunks = str_match(chunks, functional_regex)[, 1]
  functional_chunks = functional_chunks[!is.na(functional_chunks)]
  if(length(functional_chunks) > 0) {
    data = addFunctions(data, functional_chunks)
    names(data) = c(col_names, functional_chunks)
    chunks = setdiff(chunks, functional_chunks)
  }
  
  
  col_names = names(data)
  interaction_chunks = chunks
  if(length(interaction_chunks) > 0)  {
    data = addInteractions(data, interaction_chunks)
    names(data) = c(col_names, interaction_chunks)
  }
   
  return(data)
}

inspectInteractionMatrix = function(data, default = 0, showHeatmap = F) {
  interaction_matrix = matrix(default, utils.PREDICTORS_NUMBER-1, utils.PREDICTORS_NUMBER-1)
  
  base_formula <- paste(Y_LABEL, "~", paste(names(ds)[-1], collapse=" + "))
  for(i in 1:(PREDICTORS_NUMBER-1) ){
    for(j in (i+1):PREDICTORS_NUMBER){
      f = paste(base_formula," + ", names(ds)[i+1]," * ",names(ds)[j+1])
      lm=lm(f, data=ds)
      interaction_matrix[i,j-1]=summary(lm)$r.squared
    }
  }
  colnames(interaction_matrix) <- names(ds)[c(-1, -2)]
  rownames(interaction_matrix) <- names(ds)[3:(PREDICTORS_NUMBER+1)]
  
  if(showHeatmap) {
    im.showHeatmap(interaction_matrix)
  }
  
  return(interaction_matrix)
}

im.showHeatmap = function(interaction_matrix) {
  pheatmap(as.data.frame(interaction_matrix), display_numbers = T, color = colorRampPalette(c("navy", "white", "firebrick3"))(50))
}


# bestSubsetSelection <- function(data, interactions, nMSE=1000, folds=5, verbose=F) {
#   # bestSubsetByNumberOfPredictors <- regsubsets(as.formula(paste(utils.Y_LABEL, ' ~ .')), 
#   #                                              data=data, nbest = 1, nvmax=length(dsNL), 
#   #                                              intercept=TRUE, method="exhaustive")
#   
#   data = ds
#   interactions = list('X_Temperature * X_Humidity','X_MatchRelevance * X_OpposingSupportersImpact', 'X_OpposingSupportersImpact * X_RestTimeFromLastMatch','X_RestTimeFromLastMatch * X_AvgPlayerValue','X_SupportersImpact * X_OpposingSupportersImpact','X_AvgPlayerValue * X_MatchRelevance')           
#   
#   formulaChunks = paste(append(interactions, names(data)[-1]), collapse = " + ")
#   oneSidedFormula = paste('~ ', formulaChunks)
#   oneSidedFormula
#   mylm = lm(formula = as.formula(paste(utils.Y_LABEL, oneSidedFormula)), data=data, x=T, y=T)
#   options(na.action = "na.fail")
#   res = dredge(global.model=mylm, subset=as.formula(oneSidedFormula))
#   
#   bestSubsetByNumberOfPredictors <- regsubsets(x         = as.formula(paste(utils.Y_LABEL, ' ~ ', formulaChunks)), 
#                                                data      = data, 
#                                                nbest     = 1, 
#                                                nvmax     = length(data), 
#                                                intercept = TRUE, 
#                                                method    = "exhaustive")
#   
#   # Matrix of boolean that shows for the each row which predictors 
#   # provide the best combination
#   predictorsTable <- summary(bestSubsetByNumberOfPredictors)$which
#   xLabels <- colnames(predictorsTable)[-1]
#   yLabel <- utils.Y_LABEL
#   
#   predictorsNumber = dim(predictorsTable)[1]
#   bestSubsets <- vector("list", 2)
#   names(bestSubsets) = c('model', 'MSE')
#   
#   for (i in 1:predictorsNumber) {
#     ithRow <- predictorsTable[i, ]
#     formula <- reformulate(xLabels[which(ithRow[-1])], yLabel, intercept=ithRow[1])
#     if(verbose) print(formula)
#     model = lm(formula, data=data, x=T, y=T)
#     summary(model)
#     bestSubsets$model[[i]] = model
#     bestSubsets$MSE[[i]]   = mean_cvMSE(model, nMSE, folds)
#   }
#   return(bestSubsets)
# } 

bestSubsetSelection <- function (data, relationships=NULL, nMSE=1000, folds=5, method, nvmax=NULL, verbose=F) {
  switch (method,
          exhaustive={
            return(exhaustiveSubsetSelection(data, relationships, nMSE, folds, verbose))
          },
          forward={
            if(is.null(relationships)) {
              return(forwardSubsetSelection(data, 
                                            relationships=NULL, 
                                            nMSE, 
                                            folds,
                                            nvmax,
                                            verbose))
            }
            polynomial_regex = POLYNOMIAL_REGEX
            polynomial_chunks = str_match(relationships, polynomial_regex)[, 1]
            polynomial_chunks = polynomial_chunks[!is.na(polynomial_chunks)]
            
            
            functional_regex = FUNCTIONAL_REGEX
            functional_chunks = str_match(relationships, functional_regex)[, 1]
            functional_chunks = functional_chunks[!is.na(functional_chunks)]
            return(forwardSubsetSelection(data, 
                                          relationships=append(polynomial_chunks, functional_chunks), 
                                          nMSE, 
                                          folds,
                                          nvmax,
                                          verbose))
          },
          {
            stop('method not implemented')
          }
  )
}

forwardSubsetSelection = function(data, relationships, nMSE=1000, folds=5, nvmax=NULL, verbose = F) {
  
  bestSubsets = vector("list", 2)
  names(bestSubsets) = c('model', 'MSE')
  
  interactions = list()
  xlabels = colnames(data)[-1]
  
  best.formula <- paste(utils.Y_LABEL, " ~ ")
  best.rsquared = 0
  best.label = ""
  inserted.labels = list()
  inserted.interactions = list()
  i = 0
  while(!(is_empty(relationships) & is_empty(interactions) & is_empty(xlabels))) {
    i = i+1    
    for(label in append(xlabels, append(relationships, interactions))) {
      temp.formula <- paste(best.formula, label)
      temp.model <- lm(temp.formula, data=data, x=T, y=T)
      temp.rsquared <- summary(temp.model)$r.squared
      
      if(temp.rsquared > best.rsquared) { 
        best.rsquared <- temp.rsquared
        best.model <- temp.model
        best.label <- label
      }
    }
      
    if(best.label %in% xlabels){
      xlabels <- xlabels[xlabels!=best.label]
      inserted.labels = append(inserted.labels, best.label)
      for(label in inserted.labels[inserted.labels!=best.label]){
        interactions = append(interactions, paste(label,"*",best.label, sep=""))
      }
    } else if(best.label %in% interactions){
      interactions <- interactions[interactions != best.label]
    }
    bestSubsets$model[[i]] <- best.model
    bestSubsets$MSE[[i]] <- mean_cvMSE(best.model, nMSE, folds)
    best.formula <- paste(best.formula, best.label, " + ")
    if(verbose) {
      cat(i)
      cat(" ")
      print(best.formula)
    }
    if(!is.null(nvmax) && i >= nvmax) {
      return(bestSubsets)
    }
    best.rsquared=0
  }
  
  return(bestSubsets)
}

exhaustiveSubsetSelection <- function (data, relationships, nMSE, folds) {
  bestSubsets = vector("list", 2)
  names(bestSubsets) = c('model', 'MSE')
  
  ds_headers = c(names(data), relationships)
  
  combination_number = (2^utils.PREDICTORS_NUMBER)-1
  
  for(comb in 1:combination_number){
    f=paste(names(myds)[Y_index], "~")
    for(k in 1:predictors){
      if( bitwAnd( comb, 2^(k-1) ) > 0 ) {
        #PRIMA DI METTERE L'INTERAZIONE
        #SE È UN'INTERAZIONE CONTROLLO SE CI SONO GIÀ GLI ALTRI DUE NELLA FORMULA
        #A PATTO CHE LE INTERAZIONI SIANO GLI ULTIMI
        if ('*' %in% ds_headers[k]){
          #interazione trovata
          pieces = str_match(unlist(str_split(f, pattern='\\s+')),INTERACTION_REGEX)
          pieces = na.omit(pieces)[,-1]
          for (i in 1:(dim(pieces)[1])){ #for each match
            firstTerm  = pieces[i,1]
            secondTerm = pieces[i,2]
          }
        }
        
        INTERACTION_REGEX
        
        
        f=paste(f, ds_headers[k], "+")
      }
    }
    f = str_sub(f,1,nchar(f)-1) # Rimuovo l'ultimo +
    subsets[1,comb]=list(lm(f, data=myds, y=TRUE, x=TRUE))
  }
  
  lm.byIndices(data, indices, addIntercept=T)
}



ds.prettyPlot = function (data, xlabel, ylabel, title) {
  data = if(is.list(data)) unlist(data)
  
  maxMSE = getSubsetsMaxMSE(data)
  ggplotDF =  cbind(c(1:length(data)),
                    data.frame(data), 
                    rep(maxMSE, length(data)) )
  names(ggplotDF) = c("x", "y", 'z')
  
  ggplot(ggplotDF, aes(x, y)) + 
    geom_line(color="#33658a") + 
    geom_point(shape=21, color="#86bbd8", fill="#86bbd8", size=6) + 
    ggtitle(title) +
    xlab(xlabel) +
    ylab(ylabel) + 
    geom_line(aes(x, z), color="red")
}

oneStandardErrorSubset = function(bestSubsets) {
  MSEs = if(is.list(bestSubsets$MSE)) unlist(bestSubsets$MSE) else bestSubsets$MSEF
  maxMSE = getSubsetsMaxMSE
  
  for(i in 1:length(MSEs)){
    if (MSEs[i] <= maxMSE){
      print(MSEs[i])
      return (bestSubsets$model[[i]])
    }
  }
}

getSubsetsMaxMSE = function(data) {
  min(data) + sqrt(var(data))
}





