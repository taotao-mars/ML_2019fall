import csv
import random
import math
import operator
import time

start = time.time()

def loadingData(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        data = list(lines)
        for x in range(len(data)-1):
            for y in range(9):
                data[x][y] = float(data[x][y])
            if random.random() < split:
                trainingSet.append(data[x])
            else:
                testSet.append(data[x])

#euclidean
def euclideanDistance(ed1, ed2, length):
    distance = 0
    for x in range(length):
        distance += pow((ed1[x] - ed2[x]), 2)
    return math.sqrt(distance)

#Manhattan
def ManhattanDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        for y in range(x + 1, length):
            distance += (abs(data1[x] - data1[y]) + abs(data2[x] - data2[y]))
    return distance

#"compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
def cosine_similarity(data1, data2, length):
    sumxdouble, sumxy, sumydouble = 0, 0, 0
    for i in range(len(data1)):
        x = data1[i]; y = data2[i]
        sumxdouble += x*x
        sumydouble += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxdouble*sumydouble)

def findNeighbors(trainingSet, tested, k):
    distances = []
    length = len(tested)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(tested, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    Votes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in Votes:
            Votes[response] += 1
        else:
            Votes[response] = 1
    sortedVotes = sorted(Votes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def calAccuracy(testSet, predictions):
    True = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            True += 1
    return ( True / float(len(testSet))) * 100.0

def main():
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    trainingSet = []
    testSet = []
    splitdata = 0.75
    loadingData('diabetes.csv', splitdata, trainingSet, testSet)
    print 'Train set is : ' + repr(len(trainingSet))
    print 'Test set is : ' + repr(len(testSet))

    predictions= []
    k = 7
    for x in range(len(testSet)):
        neighbors = findNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('--> Predicted=' + repr(result) + ', Actual=' + repr(testSet[x][-1]))
        #confusion matrix
        if result == 1:
            if testSet[x][-1]==1:
                TP += 1

            if testSet[x][-1]==0:
                FP += 1

        if result==0:
            if testSet[x][-1]==1:
                FN += 1
            if testSet[x][-1]==0:
                TN += 1
    actualNegative = TN+FP
    actualPositive =  FN+TP
    predictedNegative= TN+FN
    predictedPositive= FP+TP
    accuracy = calAccuracy(testSet, predictions)
    print('----------------------------------------------')
    print('--> Accuracy: ' + repr(accuracy) + '%')
    print('----------------------------------------------')
    print '--> Train set is: ' + repr(len(trainingSet))
    print '--> Test set is: ' + repr(len(testSet))
    print('----------------------------------------------')
    print('--> TP:'+repr(TP))
    print('--> FP:'+repr(FP))
    print('--> TN:'+repr(TN))
    print('--> FN:'+repr(FN))
    print('----------------------------------------------')
    print('--> actualNegative:'+repr(actualNegative))
    print('--> actualPositive:'+repr(actualPositive))
    print('--> predictedNegative:'+repr(predictedNegative))
    print('--> predictedPositive:'+repr(predictedPositive))
main()

end = time.time()
print('----------------------------------------------')
print ('--> Run time: ' +repr(end - start))