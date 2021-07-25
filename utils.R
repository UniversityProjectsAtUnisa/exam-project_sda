#================ Config constants =====================

POLYNOMIAL_REGEX = '.*I\\((.*)\\^(\\d+)\\)'
FUNCTIONAL_REGEX = '(\\w*)\\((\\w*)\\)'

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

inspectLm = function(model, nMSE=1000) {
  z <- list()
  
  print("================= SUMMARY =================")
  z$summary <- summary(model)
  print(z$summary)
  
  print("==================  MSE  ==================")
  z$MSE <- mean_cvMSE(model,nMSE)
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

addNonLinearities = function(data, ...) {
  polynomial_regex = POLYNOMIAL_REGEX
  functional_regex = FUNCTIONAL_REGEX
  chunks = list(...)
  
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
