
# function that uses Rpart to make spatial clusters based on a grouping variable

makeSpatialClusters <- function(clusters, X, coordinates){
  # do the polynomial expansion of the coordinates, for feeding into Rpart
  coords <- as.data.frame(polym(x = coordinates$lat, y = coordinates$lon, degree = clusters$degree))
  # set up the target for rpart
  targ <- apply(X[,grepl(clusters$regex, colnames(X))], 1, clusters$FUN)
  # fit the tree
  classtree <- rpart(targ~., data = coords, control = rpart.control(cp = 0))
  # prune it
  classtree <- prune(classtree, cp = classtree$cptable[which.max(classtree$cptable[,"nsplit"][classtree$cptable[,"nsplit"]<clusters$n_cluster]),"CP"])
  # extract a factor vector indicating positons
  spatial_clusters <- factor(as.numeric(as.factor(classtree$where)))
  return(spatial_clusters)
}