import numpy as np




def kmeans(X, k, maxIt):
    '''
    :param X: 数据集,数据集的最后一列表示标签值（或者组号）
    :param k: k个分类
    :param maxIt: 循环几次
    :return:
    '''
    numPoints, numDim = X.shape

    dataSet = np.zeros((numPoints, numDim + 1))
    dataSet[:, :-1] = X

    # Initialize centroids randomly
    centroids = dataSet[np.random.randint(numPoints, size = k), :]
    centroids = dataSet[0:2, :]
    #Randomly assign labels to initial centorid
    centroids[:, -1] = range(1, k +1)

    # Initialize book keeping vars.
    iterations = 0
    oldCentroids = None

    # Run the main k-means algorithm
    while not shouldStop(oldCentroids, centroids, iterations, maxIt):
        print("iteration: \n", iterations)
        print ("dataSet: \n", dataSet)
        print ("centroids: \n", centroids)
        # Save old centroids for convergence test. Book keeping.
        oldCentroids = np.copy(centroids)
        iterations += 1

        # Assign labels to each datapoint based on centroids
        updateLabels(dataSet, centroids)

        # Assign centroids based on datapoint labels
        centroids = getCentroids(dataSet, k)

    # We can get the labels too by calling getLabels(dataSet, centroids)
    return dataSet

def shouldStop(oldCentroids,centroids,iterations,maxIt):
    if iterations > maxIt:
        return True
    return np.array_equal(oldCentroids,centroids)

def updateLabels(dataset,centroids):
    numPoints,numDim = dataset.shape
    #算每一行的点离哪个中心点最近
    for i in range(0,numPoints):
        dataset[i,-1] = getLabelFromClosestCentroid(dataset[i,:-1],centroids)

def getLabelFromClosestCentroid(dataRow,centroids):
    label = centroids[0,-1]
    #numpy.linalg.norm传入任意两个向量，-为距离
    minDist = np.linalg.norm(dataRow-centroids[0,:-1])
    #找最小距离
    for i in range(1,centroids.shape[0]):
        dist = np.linalg.norm(dataRow-centroids[i,:-1])
        if dist < minDist:
            minDist = dist
            label = centroids[i,-1]
    print("minDist:"+str(minDist))
    return label
def getCentroids(dataSet,k):
    result = np.zeros((k,dataSet.shape[1]))
    for i in range(1,k+1):
        #所有求 标签值为i的值
        oneCluster = dataSet[dataSet[:,-1]==i,:-1]
        #axis =0 对行求平均值，axis=1 对列求平均值
        result[i-1,:-1]=np.mean(oneCluster,axis=0)
        #最后赋值标签
        result[i-1,-1]=i
    return result

x1 = np.array([1, 1])
x2 = np.array([2, 1])
x3 = np.array([4, 3])
x4 = np.array([5, 4])
testX = np.vstack((x1, x2, x3, x4))

result = kmeans(testX,2,10)
print("final result:\n"+str(result))