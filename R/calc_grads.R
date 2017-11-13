
calc_grads<- function(plist, hlay = NULL, yhat = NULL, curBat = NULL, droplist = NULL, dropinp = NULL){
plist <- parlist
hlay <- hlayers
curBat <- droplist <- NULL
  #subset the parameters and hidden layers based on the droplist
  if (!is.null(droplist)){
    Xd <- X[,dropinp, drop = FALSE]
    if (nlayers > 1){
      #drop from parameter list emanating from input
      plist[[1]] <- plist[[1]][c(TRUE,dropinp),droplist[[1]]]
      # drop from subsequent parameter matrices
      if (nlayers>2){
        for (i in 2:(nlayers-1)){
          plist[[i]] <- plist[[i]][c(TRUE, droplist[[i-1]]), droplist[[i]], drop = FALSE]
        }
      }
      plist[[nlayers]] <- plist[[nlayers]][c(TRUE, droplist[[nlayers-1]]), 
                                           droplist[[nlayers]][(ncol(param)+1):length(droplist[[nlayers]])], 
                                           drop = FALSE]
    } else { #for one-layer networks
      #drop from parameter list emanating from input
      plist[[1]] <- plist[[1]][c(TRUE,dropinp),
                               droplist[[nlayers]][(ncol(param)+1):length(droplist[[nlayers]])], 
                               drop = FALSE]
    }
    # manage parametric/nonparametric distinction in the top layer
    plist$beta <- plist$beta[droplist[[nlayers]][(ncol(param)+1):length(droplist[[nlayers]])]]
    
  } else {Xd <- X}#for use below...  X should be safe given scope, but extra assignment is cheap here
  if (!is.null(curBat)){CB <- function(x){x[curBat,,drop = FALSE]}} else {CB <- function(x){x}}
  if (is.null(yhat)){yhat <- getYhat(plist, hlay = hlay)}
  NL <- nlayers + as.numeric(!is.null(convolutional))
  grads <- grad_stubs <- vector('list', NL + 1)
  grad_stubs[[length(grad_stubs)]] <- Matrix(2 * (y - yhat)) # top layer derivative stub
  for (i in NL:1){
    if (i == NL){outer_param = as.matrix(c(plist$beta))} else {outer_param = plist[[i+1]]}
    if (i == 1){lay = CB(Xd)} else {lay= CB(hlay[[i-1]])}
    # for temporal layers, only pass the time-varying variables
    if (names(parlist)[i] == "temporal"){
      if (!is.null(droplist)){
        stop("Dropout and convolutional are not yet compatible")
      }
      lay <- lay[, !is.na(topology)]
    }
    if (names(parlist)[i] == "spatial"){
      facdum <- model.matrix(~ clusters$spatialClusters - 1) # matrix of dummies for the spatial clusters
      TV <- t(KhatriRao(t(lay), t(facdum))) # this is repeated -- it is already done in calc_hlayers.  perhaps find a way to stash it somewhere and reuse it. 
      NTV <- Xd[, is.na(topology)]
      lay <- cbind(TV, NTV)
    }
    #add the bias
    lay <- cbind(1, lay) #add bias to the hidden layer
    if (i != NL){outer_param <- outer_param[-1,, drop = FALSE]}      #remove parameter on upper-layer bias term
    # inner term.  use eigen if not sparse
    # if ("dgcMatrix" %ni% c(unlist(class(lay)))){
    #   lay <- as.matrix(lay)
    #   inner <- eigenMapMatMult(lay, plist[[i]])
    # } else {
    #   inner <- lay %*% plist[[i]]
    # }
    inner <- MatMult(lay, plist[[i]])
    if (i == 1 & !is.null(clusters)){ #if spatial conv
      inner <- cbind(t(KhatriRao(t(inner), t(facdum))), NTV)
    }
    grad_stubs[[i]] <- inner * MatMult(grad_stubs[[i+1]], Matrix::t(outer_param))
    # ULTIMATELY THE GRAD STUB SHOULD BE 5873 X 240.  THE PROBLEM IS THAT THE OUTER PARAM IS TOO BIG.  IT NEEDS TO LOSE THE PARAMETERS ASSOCIATED WITH THE NTV VARIABLES, AND IT NEEDS TO HAVE SOME OPERATION DONE TO POOL THE SPATIAL EXPANSION
  }
  # multiply the gradient stubs by their respective layers to get the actual gradients
  # first coerce them to regular matrix classes so that the C code for matrix multiplication can speed things up
  

  
  REALLY NOT SURE ABOUT HOW THE LOWEST GRAD STUB IS MADE
IN THE FORWARD PASS, I DO 
  TV <- t(KhatriRao(t(hlayers[[i-1]]), t(facdum)))
  NTV <- X[, is.na(topology)]
  hlayers[[i]] <- activ(cbind(1, TV, NTV) %*% parlist$spatial)
IN THE BACKWARD PASS, I HAVE IT LIKE
  inner <- activ_prime(lay %*% plist[[i]])
  inner <- cbind(t(KhatriRao(t(inner), t(facdum))), NTV)
THE NTV TERM IS NOT AFFECTED BY ACTIV PRIME.
  
WRONG DIMENSIONS WHEN DOING THE GRADS
  
  grad_stubs <- lapply(grad_stubs, as.matrix)
  hlay <- lapply(hlay, as.matrix)
  for (i in 1:length(grad_stubs)){
    if (i == 1){lay = as.matrix(CB(Xd))} else {lay= CB(hlay[[i-1]])}
    if (i != length(grad_stubs) | is.null(fe_var)){# don't add bias term to top layer when there are fixed effects present
      lay <- cbind(1, lay) #add bias to the hidden layer
    }
    grads[[i]] <- eigenMapMatMult(t(lay), as.matrix(grad_stubs[[i]]))
  }
  # if using dropout, reconstitute full gradient
  if (!is.null(droplist)){
    emptygrads <- lapply(parlist, function(x){x*0})
    # bottom weights
    if (nlayers > 1){
      emptygrads[[1]][c(TRUE,dropinp),droplist[[1]]] <- grads[[1]]
      if (nlayers>2){
        for (i in 2:(nlayers-1)){
          emptygrads[[i]][c(TRUE, droplist[[i-1]]), droplist[[i]]] <- grads[[i]]
        }
      }
      emptygrads[[nlayers]][c(TRUE, droplist[[nlayers-1]]), 
                            droplist[[nlayers]][(ncol(param)+1):length(droplist[[nlayers]])]] <- grads[[nlayers]]
    } else { #for one-layer networks
      emptygrads[[1]][c(TRUE,dropinp),
                      droplist[[1]][(ncol(param)+1):length(droplist[[1]])]] <- grads[[1]]
    }
    #top-level
    emptygrads$beta <- emptygrads$beta_param <- NULL
    emptygrads[[nlayers + 1]] <- matrix(rep(0, length(parlist$beta)+length(parlist$beta_param))) #empty
    emptygrads[[nlayers + 1]][droplist[[nlayers]]] <- grads[[nlayers + 1]]
    # all done
    grads <- emptygrads
  }
  #process the gradients for the convolutional layers
  if (!is.null(convolutional)){
    if (!is.null(droplist)){
      warning("dropout not yet made to work with conv nets")
    }
    #mask out the areas not in use
    gg <- grads[[1]] * convMask
    #gradients for conv layer.  pooling via rowMeans
    grads_convParms <- foreach(i = 1:convolutional$Nconv) %do% {
      idx <- (1+N_TV_layers*(i-1)):(N_TV_layers*i)
      rowMeans(foreach(j = idx, .combine = cbind) %do% {x <- gg[,j]; x[x!=0][-1]})
    }
    grads_convBias <- foreach(i = 1:convolutional$Nconv, .combine = c) %do% {
      idx <- (1+N_TV_layers*(i-1)):(N_TV_layers*i)
      mean(gg[1,idx])
    }
    # make the layer
    convGrad <- makeConvLayer(grads_convParms, grads_convBias)
    #set the gradients on the time-invariant terms to zero
    convGrad[,(N_TV_layers * convolutional$Nconv+1):ncol(convGrad)] <- 0
    grads[[1]] <- convGrad
  }
  return(grads)
}
