import numpy as np 
from .distances import euclidean_distances, manhattan_distances

class KNearestNeighbor():    
    def __init__(self, n_neighbors, distance_measure='euclidean', aggregator='mode'):

        self.n_neighbors = n_neighbors
        self.distance_measure = distance_measure
        self.aggregator = aggregator
        self.features = None
        self.targets = None


    def fit(self, features, targets):

        self.features = features
        self.targets = targets
        

    def predict(self, features, ignore_first=False):

        labels = np.zeros((features.shape[0], self.targets.shape[1]))   #creates array of shape (n_samples, n_dimensions)

        if self.distance_measure == 'euclidean': distancesmatrix = euclidean_distances(features, self.features)
        else: distancesmatrix = manhattan_distances(features, self.features)

        rownum=0         #counts which sample being considered
        for row in distancesmatrix:
            if ignore_first == True:
                smallind = (np.argsort(row))[1:(self.n_neighbors + 1)]
            else:
                smallind = (np.argsort(row))[0:(self.n_neighbors)]

            nntar = self.targets[smallind]  #creates array of k-nearest neighbors targets

            for column in range(nntar.shape[1]):    #predict label based on aggregator
                if self.aggregator == 'mean':
                    tar = np.mean(nntar[:,column])
                elif self.aggregator == 'mode':
                    nalist = list(nntar[:,column])
                    tar = max(nalist, key=nalist.count)
                else:
                    tar = np.median(nntar[:,column])

                labels[rownum, column] = tar        #add predicted label to labels matrix

            rownum +=1

        return labels
