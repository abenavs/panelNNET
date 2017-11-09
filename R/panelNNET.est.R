panelNNET.est <-
function(y, X, hidden_units, fe_var, maxit, lam, time_var, param, parapen, parlist
         , verbose, report_interval, gravity, convtol, RMSprop
         , start.LR, activation
         , batchsize, maxstopcounter, OLStrick, initialization, dropout_hidden
         , dropout_input, convolutional, ...){

#y = dat$yield
#X = dat[,grepl('tmax|tmin|wspd|relh|radiation|lat|lon|prc|prop_irr|rotation|tillage|friability',colnames(dat))]
#param = Xp
#hidden_units = 10
#parapen = rep(0, ncol(param))
#fe_var = dat$reap
#maxit = 10
#lam = .1
#time_var = dat$year
#verbose = T
#gravity = 1.01
#convtol = 1e-5
#activation = 'lrelu'
#start_LR = .001
#parlist = NULL
#OLStrick = TRUE
#initialization = 'HZRS'
#maxit = 100
#report_interval = 5
#RMSprop = T
#start.LR <- .01
#maxstopcounter <- 10
#batchsize = nrow(X)
#dropout_hidden <- dropout_input <- 1
#datestring <- substr(colnames(X), nchar(colnames(X))-4, nchar(colnames(X)))
#topology <- as.POSIXlt(datestring, format = "%m_%d")$yday
#convolutional <- list(Nconv = 5, 
#                      span = 5, 
#                      step = 5, 
#                      topology = topology
#                      )
#coordinates <- dat[, c("lat", "lon")]
#clusters <- list(regex = "tmin04",
#                 clusters = NULL,
#                 FUN = mean,
#                 degree = 3,
#                 n_cluster = 4
#)




# X <- data.frame(v1 = rnorm(10), v2 = rnorm(10), v3 = rnorm(10))
# fac <- c(rep("A",2), rep("B", 4), rep("C", 4))
# facdum <- model.matrix(~fac-1)
# t(KhatriRao(t(X), t(facdum)))

  
  ##########
  #Define internal functions
  getYhat <- function(pl, hlay = NULL){ 
    #Update hidden layers
    if (is.null(hlay)){hlay <- calc_hlayers(pl,
                                            X = X,
                                            param = param,
                                            fe_var = fe_var,
                                            nlayers = nlayers,
                                            convolutional = convolutional,
                                            activ = activation)}
    #update yhat
    if (!is.null(fe_var)){
      Zdm <- demeanlist(as.matrix(hlay[[length(hlay)]]), list(fe_var))
      fe <- (y-ydm) - (as.matrix(hlay[[length(hlay)]])-Zdm) %*% as.matrix(c(pl$beta_param, pl$beta))
      yhat <- hlay[[length(hlay)]] %*% c(pl$beta_param, pl$beta) + fe    
    } else {
      yhat <- hlay[[length(hlay)]] %*% c(pl$beta_param, pl$beta)
    }
    return(as.numeric(yhat))
  }

  makeConvLayer <- function(convParms, convBias){
    # time-varying portion
    TV <- foreach(i = 1:convolutional$Nconv, .combine = cbind) %do% {
      apply(convMask[,1:N_TV_layers], 2, function(x){# this assumes that the feature detectors have identical shapes
        x[x!=0] <- convParms[[i]]
        x <- c(convBias[i], x)
        return(x)
      })
    }
    # NTV <- convMask[,colnames(convMask) %ni% convolutional$topology]
    # return(Matrix(cbind(TV, NTV)))
    return(Matrix(TV))
  }
  
  ###########################
  # start fitting
  ###########################
  # do scaling
  X <- scale(X)
  if (!is.null(param)){
    param <- scale(param)
  }
  if (activation == 'tanh'){
    activ <- tanh
    activ_prime <- tanh_prime
  }
  if (activation == 'logistic'){
    activ <- logistic
    activ_prime <- logistic_prime
  }
  if (activation == 'relu'){
    activ <- relu
    activ_prime <- relu_prime
  }
  if (activation == 'lrelu'){
    activ <- lrelu
    activ_prime <- lrelu_prime
  }
  nlayers <- length(hidden_units)
  # initialize the convolutional layer, if present
  if (!is.null(convolutional)){
    # set set the topology to start at 1, if it isn't already there.  give a warning if it isn't.
    if (min(convolutional$topology, na.rm =T)>1){
      convolutional$topology <- convolutional$topology - min(convolutional$topology, na.rm =T) +1 
      print("minimum value in supplied topology greater than 1.  subtracting to get it to start at 1.")
      warning("minimum value in supplied topology greater than 1.  subtracting to get it to start at 1.")
    }
    # make the convolutional masking matrix if using conv nets
    convMask <- convolutional$convmask <- makeMask(X, convolutional$topology, convolutional$span, convolutional$step, convolutional$Nconv)
    # store the number of time-varying variables
    # both in the local env for convenience, and in the convolutional object for passing to other functions
    N_TV_layers <- convolutional$N_TV_layers <- sum(unique(colnames(convMask)) %in% convolutional$topology)
    # For each convolutional "column", initialize the single parameter vector that will be shared among columns
    if (is.null(convolutional$convParms)){
      convParms <- convolutional$convParms <- foreach(i = 1:convolutional$Nconv) %do% {
        rnorm(sum(convMask[,1]), sd = 2/sqrt(sum(convMask[,1])))
      }
    }
    # Initialize convolutional layer bias, if not present
    # new version: bias terms are not individual to each span, but shared by each span
    if (is.null(convolutional$convBias)){
      convBias <- rnorm(convolutional$Nconv, sd = 2/sqrt(sum(convMask[,1])))
    }
    # initialize the convolutional parlist, if not present
    if (is.null(convolutional$convParMat)){
      convParMat <- convolutional$convParMat <- makeConvLayer(convParms, convBias)
    }
  }
  # compute the spatial clusters, if not NULL
  if (!is.null(clusters)){
    # make sure that convolutonal isn't NULL
    if (is.null(convolutional)){
      stop("spatial conv nets without temporal convolution aren't supported yet")
    }
    spatialClusters <- clusters$spatialClusters <- makeSpatialClusters(clusters, X, coordinates)
    # this can be removed -- it is handled in the chunk below that handles the parlist generation
    # # initialize the spatial convolution parameters
    # nsp <- length(unique(spatialClusters)) * #number of clusters, times...
    #         ncol(convMask) * # number of time-varying variables, times...
    #         hidden_units[1]
    # sp_param <- matrix(rnorm(nsp, sd = 2/sqrt(nsp)), ncol = hidden_units[1])
  }
  #get starting weights, either randomly or from a specified parlist
  if (is.null(parlist)){#random starting weights
    # start with the fully-connected, hidden units
    parlist <- vector('list', nlayers)
    for (i in 1:nlayers){
      if (i == 1){
        if (is.null(convolutional)){
          D <- ncol(X)
        } else {
          if (is.null(clusters)){
            D <- ncol(convolutional$convParMat)
          } else {
            # if spatially convoluted, dimension is number of spatial clusters times number of conv layers, 
            # plus number of time-invariant variables
            D <- length(unique(spatialClusters)) * ncol(convMask) + sum(is.na(topology)) 
          }
        }
      } else {
        D <- hidden_units[i-1]
      }
      if (initialization %ni% c('XG', 'HZRS')){#random initialization schemes
        ubounds <- .7 #follows ESL recommendaton
      } else {
        if (initialization == 'XG'){
          ubounds <- sqrt(6)/sqrt(D+hidden_units[i]+2)#2 is for the bias.  Not sure why 2.  Would need to go back and read the paper.  
        }
        if (initialization == 'HZRS'){
          ubounds <- 2*sqrt(6)/sqrt(D+hidden_units[i]+2)#2 is for the bias.  Not sure why 2.  Would need to go back and read the paper.
        }
      }
      parlist[[i]] <- matrix(runif((hidden_units[i])*(D+1), -ubounds, ubounds), ncol = hidden_units[i])
    }
    # vector of parameters at the top layer
    parlist$beta <- runif(hidden_units[i], -ubounds, ubounds)
    # parameters on parametric terms
    if (is.null(param)){
      parlist$beta_param <-  NULL
    } else {
      parlist$beta_param <- runif(ncol(param), -ubounds, ubounds)
    }
    #add the bias term/intercept onto the front, if there are no FE's
    parlist$beta_param <- c(runif(is.null(fe_var), -ubounds, ubounds), parlist$beta_param)
    # initialize the spatial convolution parameters
    if (!is.null(clusters)){
      names(parlist)[[1]] <- "spatial"
    }
    # add convolutional layer(s) on the bottom
    if (!is.null(convolutional)){
      parlist <- c(convolutional$convParMat, parlist)
      names(parlist)[[1]] <- "temporal"
    }

    #if there are no FE's, append a 0 to the front of the parapen vec, to leave the intercept unpenalized
    if(is.null(fe_var)){
      parapen <- c(0, parapen)
    }
  }
  #compute hidden layers given parlist
  hlayers <- calc_hlayers(parlist, X = X, param = param, 
                          fe_var = fe_var, nlayers = nlayers, 
                          convolutional = convolutional, activation = activation,
                          clusters = clusters)
  #calculate ydm and put it in global...
  # note oct 12: I don't remember why this is in global.  Maybe try moving back to <-?
  if (!is.null(fe_var)){
    ydm <<- demeanlist(y, list(fe_var)) 
  }
  #####################################
  #start setup
  #get starting mse
  yhat <- as.numeric(getYhat(parlist, hlay = hlayers))
  mse <- mseold <- mean((y-yhat)^2)
  pl_for_lossfun <- parlist[!grepl('beta', names(parlist))]
  if (!is.null(convolutional)){
    pl_for_lossfun[[1]] <- unlist(c(convolutional$convParms, convolutional$convBias))
  }
  loss <- mse + lam*sum(c(parlist$beta_param*parapen 
    , parlist$beta
    , unlist(sapply(pl_for_lossfun, as.numeric)))^2
  )
  LRvec <- LR <- start.LR# starting step size
  #Calculate gradients
  grads <- calc_grads(parlist, hlayers, yhat, droplist = NULL, dropinp = NULL)
  #Initialize updates
  updates <- lapply(parlist, function(x){x*0})
  #initialize G2 term for RMSprop
  if (RMSprop == TRUE){
    #Prior gradients are zero at first iteration...
    G2 <- lapply(parlist, function(x){x*0})
    #squashing all of the numeric list elements into a matrix/vector
    betas <- matrix(unlist(G2[grepl('beta', names(G2))]))
    G2 <- G2[!grepl('beta', names(G2))]
    G2[[length(G2)+1]] <- betas
  } else {G2 <- NULL}
  # initialize terms used in the while loop
  D <- 1e6
  stopcounter <- iter <- 0
  msevec <- lossvec <- c()
  ###############
  #start iterating
  while(iter < maxit & stopcounter < maxstopcounter){
    oldpar <- list(parlist=parlist, hlayers=hlayers, grads=grads
      , yhat = yhat, mse = mse, mseold = mseold, loss = loss, updates = updates, G2 = G2
      , msevec = msevec, lossvec = lossvec)
    #Start epoch
    #Assign batches
    batchid <- sample(1:nrow(X)%/%batchsize +1)
    if (min(table(batchid))<(batchsize/2)){#Deal with orphan batches
      batchid[batchid == max(batchid)] <- sample(1:(max(batchid) - 1), min(table(batchid)), replace = TRUE)
    }
    for (bat in 1:max(batchid)) { # run minibatch
      curBat <- which(batchid == bat)
      hlay <- hlayers#h lay may have experienced dropout, as distinct from hlayers
      # if using dropout, generate a droplist
      if (dropout_hidden < 1){
        droplist <- lapply(hlayers, function(x){
          todrop <- as.logical(rbinom(ncol(x), 1, dropout_hidden))
          if (all(todrop==FALSE)){#ensure that at least one unit is present
            todrop[sample(1:length(todrop))] <- TRUE
          }
          return(todrop)
        })
        # remove the parametric terms from dropout contention
        droplist[[nlayers]][1:ncol(param)] <- TRUE
        # dropout from the input layer
        todrop <- rbinom(ncol(X), 1, dropout_input)
        if (all(todrop==FALSE)){# ensure that at least one unit is present
          todrop[sample(1:length(todrop))] <- TRUE
        }
        dropinp <- as.logical(todrop)
        for (i in 1:nlayers){
          hlay[[i]] <- hlay[[i]][,droplist[[i]], drop = FALSE]
        }
        Xd <- X[,dropinp]
      } else {Xd <- X; droplist = NULL}
      # before updating gradients, compute square of gradients for RMSprop
      if (RMSprop ==  TRUE){oldG2 <- lapply(grads, function(x){.9*x^2})} #old G2 term 
      # Get updated gradients
      grads <- calc_grads(plist = parlist, hlay = hlay
        , yhat = yhat[curBat], curBat = curBat, droplist = droplist, dropinp = dropinp)
      # Calculate updates to parameters based on gradients and learning rates
      if (RMSprop == TRUE){
        newG2 <- lapply(grads, function(x){.1*x^2}) #new gradient is squared and multiplied by .1
        G2 <- mapply('+', newG2, oldG2)
        # updates to beta
        uB <- LR/sqrt(G2[[length(G2)]]+1e-10) * grads[[length(grads)]]
        updates$beta_param <- uB[1:length(parlist$beta_param)]
        updates$beta <- uB[ncol(param)+(1:length(parlist$beta))]
        # updates to lower layers
        NL <- nlayers + as.numeric(!is.null(convolutional))
        for(i in NL:1){
          updates[[i]] <- LR/sqrt(G2[[i]]+1e-10) * grads[[i]]
        }
      } else { #if RMSprop == FALSE
        uB <- LR * grads[[length(grads)]]
        updates$beta_param <- uB[1:length(parlist$beta_param)]
        updates$beta <- uB[ncol(param)+(1:length(parlist$beta))]
        NL <- nlayers + as.numeric(!is.null(convolutional))
        for(i in NL:1){
          updates[[i]] <- LR * grads[[i]]
        }
      }
      # weight decay
      if (lam != 0) {
        wd <- lapply(parlist, function(x){x*lam*LR})
        updates <- mapply("+", updates, wd)
        # don't update the pass-through weights for the non-time-varying variables when using conv 
        if (!is.null(convolutional)){
          updates[[1]][,colnames(updates[[1]]) %ni% convolutional$topology] <- 0
        }
      }
      # Update parameters from update list
      parlist <- mapply('-', parlist, updates)
      # Update hidden layers
      hlayers <- calc_hlayers(parlist, X = X, param = param, fe_var = fe_var, 
                              nlayers = nlayers, convolutional = convolutional, activ = activation)
      # OLS trick!
      if (OLStrick == TRUE){
        parlist <- OLStrick_function(parlist = parlist, hidden_layers = hlayers, y = y
          , fe_var = fe_var, lam = lam, parapen = parapen)
      }
      #update yhat
      yhat <- getYhat(parlist, hlay = hlayers)
      mse <- mean((y-yhat)^2)
      msevec <- append(msevec, mse)
      pl_for_lossfun <- parlist[!grepl('beta', names(parlist))]
      if (!is.null(convolutional)){ # coerce the convolutional parameters to a couple of vectors to avoid double-counting in the loss
        convolutional$convParms <- foreach(i = 1:convolutional$Nconv) %do% {
          idx <- (1+N_TV_layers*(i-1)):(N_TV_layers*i)
          rowMeans(foreach(j = idx, .combine = cbind) %do% {x <- pl_for_lossfun[[1]][,j]; x[x!=0][-1]})
        }
        convolutional$convBias <- foreach(i = 1:convolutional$Nconv, .combine = c) %do% {
          idx <- (1+N_TV_layers*(i-1)):(N_TV_layers*i)
          mean(pl_for_lossfun[[1]][1,idx])
        }
        pl_for_lossfun[[1]] <- c(unlist(convolutional$convParms, convolutional$convBias))
      }
      loss <- mse + lam*sum(c(parlist$beta_param*parapen
                              , parlist$beta
                              , unlist(sapply(pl_for_lossfun, as.numeric)))^2
      )
      lossvec <- append(lossvec, loss)
    } #finishes epoch

    #Finished epoch.  Assess whether MSE has increased and revert if so
    mse <- mean((y-yhat)^2)
    loss <- mse + lam*sum(c(parlist$beta_param*parapen
                            , parlist$beta
                            , unlist(sapply(pl_for_lossfun, as.numeric)))^2
    )
    #If loss increases...
    if (oldpar$loss <= loss){
      parlist <- oldpar$parlist
      updates <- oldpar$updates
      G2 <- oldpar$G2
      hlayers <- oldpar$hlayers
      grads <- oldpar$grads
      yhat <- oldpar$yhat
      mse <- oldpar$mse
      stopcounter <- stopcounter + 1
      loss <- oldpar$loss
      msevec <- oldpar$msevec
      lossvec <- oldpar$lossvec
      LR <- LR/2
      if(verbose == TRUE){
        print(paste0("Loss increased.  halving LR.  Stopcounter now at ", stopcounter))
      }
    } else { # if loss doesn't increase
      LRvec[iter+1] <- LR <- LR*gravity      #gravity...
      D <- oldpar$loss - loss
      if (D < convtol){
        stopcounter <- stopcounter +1
        if(verbose == TRUE){print(paste('slowing!  Stopcounter now at ', stopcounter))}
      } else { # reset stopcounter if not slowing per convergence tolerance
        stopcounter <-0
      }
      if  (verbose == TRUE & iter %% report_interval == 0){
        writeLines(paste0(
          "*******************************************\n"
          , 'Lambda = ',lam, "\n"
          , "Hidden units -> ",paste(hidden_units, collapse = ' '), "\n"
          , " Batch size is ", batchsize, " \n"
          , " Completed ", iter, " epochs. \n"
          , " Completed ", bat, " batches in current epoch. \n"
          , "mse is ",mse, "\n"
          , "last mse was ", oldpar$mse, "\n"
          , "difference is ", oldpar$mse - mse, "\n"
          , "loss is ",loss, "\n"
          , "last loss was ", oldpar$loss, "\n"
          , "difference is ", oldpar$loss - loss, "\n"
          , "input layer dropout probability: ", dropout_input, "\n"
          , "hidden layer dropout probability: ", dropout_hidden, "\n"
          , "*******************************************\n"  
        ))
        par(mfrow = c(3,2))
        plot(y, yhat, col = rgb(1,0,0,.5), pch = 19, main = 'in-sample performance')
        abline(0,1)
        plot(LRvec, type = 'b', main = 'learning rate history')
        plot(msevec, type = 'l', main = 'all epochs', ylim = range(c(msevec), na.rm = TRUE))
        plot(msevec[(1+(iter)*max(batchid)):length(msevec)], type = 'l', ylab = 'mse', main = 'Current epoch')
        plot(lossvec, type = 'l', main = 'all epochs')
        plot(lossvec[(1+(iter)*max(batchid)):length(lossvec)], type = 'l', ylab = 'loss', main = 'Current epoch')
      } # fi verbose 
    } # fi if loss increases 
    iter <- iter+1
  } #closes the while loop
  #If trained with dropput, weight the layers by expectations
  if(dropout_hidden<1){
    for (i in nlayers:1){
      if (i == 1){
        parlist[[i]] <- parlist[[i]] * dropout_input
      } else {
        parlist[[i]] <- parlist[[i]] * dropout_hidden
      }
    }
    parlist$beta <- parlist$beta * dropout_hidden
    if (OLStrick == TRUE){
      parlist <- OLStrick_function(parlist = parlist, hidden_layers = hlayers, y = y
        , fe_var = fe_var, lam = lam, parapen = parapen)
    }
    #redo the hidden layers based on the new parlist
    hlayers <- calc_hlayers(parlist, X = X, param = param,
                            fe_var = fe_var, nlayers = nlayers,
                            convolutional = convolutional, activ = activation)
    yhat <- getYhat(parlist, hlay = hlayers)
  }
  conv <- (iter < maxit)#Did we get convergence?
  if(is.null(fe_var)){
    fe_output <- NULL
  } else {
    Zdm <- demeanlist(as.matrix(hlayers[[length(hlayers)]]), list(fe_var))
    Zdm <- Matrix(Zdm)
    fe <- (y-ydm) - as.matrix(hlayers[[length(hlayers)]]-Zdm) %*% as.matrix(c(
        parlist$beta_param, parlist$beta
    ))
  fe_output <- data.frame(fe_var, fe)
  }
  output <- list(yhat = yhat, parlist = parlist, hidden_layers = hlayers
    , fe = fe_output, converged = conv, mse = mse, loss = loss, lam = lam, time_var = time_var
    , X = X, y = y, param = param, fe_var = fe_var, hidden_units = hidden_units, maxit = maxit
    , final_improvement = D, msevec = msevec, RMSprop = RMSprop, convtol = convtol
    , grads = grads, activation = activation, parapen = parapen
    , batchsize = batchsize, initialization = initialization, convolutional = convolutional
    , dropout_hidden = dropout_hidden, dropout_input = dropout_input)
  return(output) # list 
}






