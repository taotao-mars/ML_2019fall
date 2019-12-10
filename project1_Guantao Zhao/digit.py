import csv
import random
import math
import operator
import time

start = time.time()


def loadingdata(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'rU') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(783):
                dataset[x][y] = int(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


def euclideanDistance(d1, d2, length):
    distance = 0
    for x in range(length):
        distance += pow((float(d1[x]) - float(d2[x])), 2)
    return math.sqrt(distance)


def findNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    Votes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][0]
        if response in Votes:
            Votes[response] += 1
        else:
            Votes[response] = 1
    sortedVotes = sorted(Votes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def calAccuracy(testSet, predictions):
    True = 0
    for x in range(len(testSet)):
        if testSet[x][0] == predictions[x]:
            True += 1
    return (int(True)/ float(len(testSet))) * 100.0


def main():
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    predictions = []
    trainingSet = []
    testSet = []
    split = 0.9998
    loadingdata('train.csv', split, trainingSet, testSet)
    print 'Train set is: ' + repr(len(trainingSet))
    print 'Test set is: ' + repr(len(testSet))

    k = 7
    for x in range(len(testSet)):
        neighbors = findNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('--> Predicted =' + repr(result) + ', Actual =' + repr(testSet[x][0]))
        # draw table for number 1
        if result == 1:
            if testSet[x][0] == 1:
                TP += 1

            if testSet[x][0] != 1:
                FP += 1

        if result != 1:
            if testSet[x][0] == 1:
                FN += 1
            if testSet[x][0] != 1:
                TN += 1

    actualNegative = TN + FP
    actualPositive = FN + TP
    predictedNegative = TN + FN
    predictedPositive = FP + TP
    accuracy_1= (float(TP)/float(actualNegative+actualPositive)) * 100
    accuracy = calAccuracy(testSet, predictions)
    print('----------------------------------------------')
    print('--> Total Accuracy: ' + repr(accuracy) + '%')
    print('----------------------------------------------')
    print '--> Train set: ' + repr(len(trainingSet))
    print '--> Test set: ' + repr(len(testSet))
    print('----------------------------------------------')
    print('--> TP:' + repr(TP))
    print('--> FP:' + repr(FP))
    print('--> TN:' + repr(TN))
    print('--> FN:' + repr(FN))
    print('----------------------------------------------')
    print('--> actualNegative:' + repr(actualNegative))
    print('--> actualPositive:' + repr(actualPositive))
    print('--> predictedNegative:' + repr(predictedNegative))
    print('--> predictedPositive:' + repr(predictedPositive))
    print('--> accuracy of #1:' + repr(accuracy_1)+ '%')

main()

end = time.time()
print('----------------------------------------------')
print ('--> Run time: ' + repr(end - start))