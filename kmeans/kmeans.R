# ---------------------------------------------
#                 K - MEANS
# ---------------------------------------------

#algoritmo k-means
kmeans <- function(dataset, n_clusters = 3, threshold = 1e-3){
  
  dataset = as.matrix(dataset)
  
  #selecionar centroides aleatoriamente
  ids = sample(1:nrow(dataset), size = n_clusters)
  centroids = dataset[ids, ]
  
  #vetor que contem para cada ponto => centroide mais prox.
  closest = rep(0, nrow(dataset))
  
  error = 2 * threshold
  
  while(error >= threshold){
    
    error = 0
    
    #calcular a distancia de cada ponto a cada centroide
    for(i in 1:nrow(dataset)){
      row = dataset[i, ]
      
      euclidean = apply(centroids, 1, function(cent){
        sqrt(sum((row - cent)^2))
      })
      
      #selecionar o centroide mais próximo do ponto
      id = which.min(euclidean)[1]
      closest[i] = id
    }
    
    #adaptacao do centroide
    old_centroids = centroids
    
    for(j in 1:n_clusters){
      ids = which(closest == j)
      centroids[j] = colMeans(dataset[ids, ])
    }
    
    #calculo do erro
    error = round(sqrt(sum((centroids - old_centroids)^2)), digits = 5)
    cat('error: ', error, ' ... \n')
  }
  
  #resultados
  ret = list()
  ret$centroids = centroids
  ret$error = error
  
  return(ret)
}

#iris: 150 amostas
kmeans.toy <- function(){
  dataset = iris[, 1:4]
  res = kmeans(dataset, n_clusters = 3, threshold = 1e-3)
  plot(dataset[, 1:2])
  points(res$centroids[, 1:2], col = 5, pch = 19, cex = 1.5)
  
  return(res)
}