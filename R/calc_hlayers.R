
calc_hlayers <- function(parlist, X = X, param = param, fe_var = fe_var, 
                         nlayers = nlayers, convolutional, activation, clusters = NULL){
  if (activation == 'tanh'){
    activ <- tanh
  }
  if (activation == 'logistic'){
    activ <- logistic
  }
  if (activation == 'relu'){
    activ <- relu
  }
  if (activation == 'lrelu'){
    activ <- lrelu
  }
  hlayers <- vector('list', nlayers)
  # number of layers
  NL <- nlayers
  if (!is.null(convolutional)){NL <- NL+1}
  for (i in 1:NL){
    if (!is.null(convolutional) & i < 3){ # i will be 3 or more whn moving past convolutions
      if (i == 1){
        D <- cbind(1, X[, !is.na(topology)])
        hlayers[[i]] <- activ(D %*% parlist$temporal)
      } else { # implicitly if i == 2
        if (!is.null(clusters)){ #if clusters, do the KRexpansion, and then tack on the time-invariant terms
          facdum <- model.matrix(~ clusters$spatialClusters - 1) # matrix of dummies for the spatial clusters
          TV <- t(KhatriRao(t(hlayers[[i-1]]), t(facdum)))
          NTV <- X[, is.na(topology)]
          hlayers[[i]] <- activ(cbind(1, TV, NTV) %*% parlist$spatial)
        } else { # if no spatial convolution
          TV <- hlayers[[i-1]]
          NTV <- X[, is.na(topology)]
          hlayers[[i]] <- activ(eigenMapMatMult(as.matrix(cbind(1, TV, NTV)), as.matrix(parlist[[i]])))
        }
      }
    } else { # if not convolutional OR i > 3
      D <- cbind(1, hlayers[[i-1]])
      if ("dgcMatrix" %in% c(unlist(class(D)), unlist(class(parlist[[i]])))){ #if/else sparse, Matrix vs RcppEigen
        hlayers[[i]] <- activ(D %*% parlist[[i]])
      } else {
        if ("dgeMatrix" %in% c(unlist(class(D)))){D <- as.matrix(D)}
        hlayers[[i]] <- activ(eigenMapMatMult(D, parlist[[i]]))
      }
    }
  }
  colnames(hlayers[[i]]) <- paste0('nodes',1:ncol(hlayers[[i]]))
  if (!is.null(param)){#Add parametric terms to top layer
    hlayers[[i]] <- cbind(param, hlayers[[i]])
    colnames(hlayers[[i]])[1:ncol(param)] <- paste0('param',1:ncol(param))
  }
  if (is.null(fe_var)){#add intercept if no FEs
    hlayers[[i]] <- cbind(1, hlayers[[i]])
  }
  return(hlayers)
}

