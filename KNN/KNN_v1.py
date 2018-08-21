import operator
import numpy as np

def createDataset():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    lables = ['A','A','B','B']
    return group, lables

def classify(input_vector, dataSet, labels, K):
    # compute the distance of input vector to all data
    diffMat = np.tile(input_vector,(dataSet.shape[0],1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = np.sqrt(sqDistances)
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(K):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def test():
    group, labels = createDataset()
    print classify([0.1,0.1], group, labels, 3)

if __name__ == '__main__':
    test()