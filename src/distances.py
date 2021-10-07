import numpy as np 

def euclidean_distances(X, Y):
    euclidean = np.zeros((X.shape[0], Y.shape[0]))
    #print(X.shape[0])
    x_ind = 0
    for row_x in X:
        y_ind = 0
        for row_y in Y:
            euclidean[x_ind,y_ind] = np.sqrt((np.square(row_x - row_y)).sum())
            y_ind +=1
        x_ind += 1

    return euclidean


def manhattan_distances(X, Y):
    manhattan = np.zeros((X.shape[0], Y.shape[0]))
    x_ind = 0
    for row_x in X:
        y_ind = 0
        for row_y in Y:
            manhattan[x_ind,y_ind] = (np.absolute(row_x - row_y)).sum()
            y_ind += 1
        x_ind += 1

    return manhattan


