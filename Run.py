import Utils
from Test import MismatchTree
import numpy as np
from sklearn import svm

K = 4
M = 1
print("K,M ", K, M)
weightsFileName = "weights-" + str(K)+str(M)

# dataLocation = "/s/chopin/a/grad/akashsht/Downloads/fisher-scop-data/"
dataLocation = "/Users/akash/Downloads/fisher-scop-data/"

def getData_1_1():
    data = Utils.loadSeqFile(path = dataLocation + "1.1/1.1.1.2/1.1.1.1:d1myt__.pos-train.seq")
    data.extend(Utils.loadSeqFile(path = dataLocation + "1.1/1.1.1.2/1.1.1.1:d1spga_.pos-train.seq"))
    # data.extend(Utils.loadSeqFile(path = dataLocation + "1.1/1.1.1.2/1.1.1.1:d3sdha_.pos-train.seq"))
    # data.extend(Utils.loadSeqFile(path = dataLocation + "1.1/1.1.1.2/1.1.1.1:d1ash__.pos-train.seq"))
    # data.extend(Utils.loadSeqFile(path = dataLocation + "1.1/1.1.1.2/1.1.1.1:d1baba_.pos-train.seq"))
    # data.extend(Utils.loadSeqFile(path = dataLocation + "1.1/1.1.1.2/1.1.1.1:d1flp__.pos-train.seq"))
    # data.extend(Utils.loadSeqFile(path = dataLocation + "1.1/1.1.1.2/1.1.1.1:d1hbg__.pos-train.seq"))
    # data.extend(Utils.loadSeqFile(path = dataLocation + "1.1/1.1.1.2/1.1.1.1:d1hdsa_.pos-train.seq"))
    # data.extend(Utils.loadSeqFile(path = dataLocation + "1.1/1.1.1.2/1.1.1.1:d1hlb__.pos-train.seq"))
    # data.extend(Utils.loadSeqFile(path = dataLocation + "1.1/1.1.1.2/1.1.1.1:d1itha_.pos-train.seq"))
    # data.extend(Utils.loadSeqFile(path = dataLocation + "1.1/1.1.1.2/1.1.1.1:d1lh1__.pos-train.seq"))
    # data.extend(Utils.loadSeqFile(path = dataLocation + "1.1/1.1.1.2/1.1.1.1:d1mba__.pos-train.seq"))
    # data.extend(Utils.loadSeqFile(path = dataLocation + "1.1/1.1.1.2/1.1.1.1:d1mbd__.pos-train.seq"))
    # data.extend(Utils.loadSeqFile(path = dataLocation + "1.1/1.1.1.2/1.1.1.1:d1myt__.pos-train.seq"))
    data.extend(Utils.loadSeqFile(path = dataLocation + "1.1/1.1.fold0.neg-train.seq"))
    data.extend(Utils.loadSeqFile(path = dataLocation + "1.1/1.1.fold1.neg-train.seq"))
    return np.asarray(data)

def getDummy():
    data = Utils.loadSeqFile(path="data/3.73.fold1.neg-train.seq")
    return np.asarray(data)


data = getData_1_1()
labels = data[:,0].astype(int)
unique, counts = np.unique(labels, return_counts=True)
print (np.asarray((unique, counts)).T)
compute = MismatchTree(debug=True)
kernelMat, leafIndex, kmerIDs, reverseIndex, directIndex = compute.makeKernel(data, K, M)
print("kernelMatrix ", kernelMat)



classifier = svm.SVC(kernel='precomputed', C=10)
classifier.fit(kernelMat, labels)
weights = compute.fastTraverse(classifier, leafIndex, reverseIndex, kmerIDs, directIndex)
np.save(weightsFileName, weights)
print(weightsFileName)
