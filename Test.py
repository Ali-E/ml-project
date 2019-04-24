import Utils
from collections import Counter
import numpy as np
from sklearn import svm
import copy

ALPHABET = ['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V' ]
K = 3
M = 1

class Node:
    def __init__(self, parent, depth):
        self.parent = parent
        self.nodeData = []
        self.children = []
        self.depth = depth
        self.word = ""
        self.number = 0

    def addNodeData(self, nodeData):
        self.nodeData.append(nodeData)

    def addChildren(self, child):
        self.children.append(child)
        

class NodeData:
    def __init__(self, kmerId, mismatch):
        self.mismatch = mismatch
        self.kmerId = kmerId
        
    def clone(self):
        nodeDataCopy = NodeData(self.kmerId, self.mismatch)
        return nodeDataCopy
        
    def __str__(self):
        return str(self.kmerId) + ": " + str(self.mismatch)


class Occurance:
    # occurance of k-mer in data #id, with #frequency
    def __init__(self, id, frequency):
        self.id = id
        self.frequency = frequency

    def __str__(self):
        return str(self.id) + " " + str(self.frequency)
        
        
class KmersID:
    def __init__(self):
        self.kmerIndex = dict()
        self.id = 0
    
    def addKmer(self, kmer):
        self.kmerIndex[self.id] = kmer
        self.id += 1
        
class MismatchTree:
    def __init__(self, debug = False):
        self.debug = debug

    def log(self, string):
        if self.debug:
            print(string)

    def mergeReverseIndexWithOld(self, reverseIndex, id, listOfKmers, kmerIDs):
        counterDictionary = Counter(listOfKmers)
        for key, value in counterDictionary.items():
            if key in reverseIndex:
                reverseIndex.get(key).append(Occurance(id, value))
            else:
                kmerIDs.addKmer(key)
                reverseIndex[key] = [Occurance(id,value)]

    def buildReverseIndex(self, data, K):
        kmerIDs = KmersID()
        reverseIndex = dict()
        for i in range(len(data)):
            read = data[i]
            listOfKmers = Utils.getKMer(read[1], K) # What is K? We should test it on different Ks like the paper
            self.mergeReverseIndexWithOld(reverseIndex, i, listOfKmers, kmerIDs)
        return reverseIndex, kmerIDs

    def dfsOnTree(self, root, K, maxMismatch, leafIndex, kmerIDs):
        if root.depth == K:
            leafIndex[root.word] = root
            return
        for char in ALPHABET:
            newChild = Node(root, root.depth+1)
            newChild.word = root.word + char
            for nodeData in root.nodeData:
                newNodeData = nodeData.clone()
                if kmerIDs.kmerIndex.get(newNodeData.kmerId)[root.depth] != char:
                    newNodeData.mismatch += 1
                if newNodeData.mismatch <= maxMismatch:
                    newChild.addNodeData(newNodeData)
            if len(newChild.nodeData) > 0:
                self.dfsOnTree(newChild, K, maxMismatch, leafIndex, kmerIDs)

    def makeLeaves(self, data, K, maxMismatch):
        leafIndex = dict()
        reverseIndex, kmerIDs = self.buildReverseIndex(data, K)
        self.log("Done Reading")
        # allKmers = reverseIndex.keys()
        root = Node(None, 0)
        for kmerId in range(len(kmerIDs.kmerIndex)):
            approvedKmer = NodeData(kmerId, 0)
            root.addNodeData(approvedKmer)
        self.log("Starting DFS ....")
        self.dfsOnTree(root, K, maxMismatch, leafIndex, kmerIDs)
        return leafIndex, kmerIDs, reverseIndex


    def normalize(self, kernelMatrix):
        N = len(kernelMatrix)
        newKernelMatrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                newKernelMatrix[i][j] = kernelMatrix[i][j]/(kernelMatrix[i][i]*kernelMatrix[j][j])

        return newKernelMatrix



    def makeKernel(self, data, K, maxMismatch):
        N = len(data)
        self.log("Making kernel with data " + str(N))
        kernelMatrix = np.zeros((N,N))
        leafIndex, kmerIDs, reverseIndex = self.makeLeaves(data, K, maxMismatch)
        self.log("leafIndex: " + str(len(leafIndex)) + " ")
        directIndex = dict();

        for key, leaf in leafIndex.items():
            occurances = dict() # { dataNum: frequency_of_that_kmer_with_mismatch}
            for node in leaf.nodeData:
                kmer = kmerIDs.kmerIndex.get(node.kmerId)
                for occurance in reverseIndex.get(kmer):

                    if occurance.id in directIndex:
                        directIndex[occurance.id].append(key)
                    else:
                        directIndex[occurance.id] = [key]

                    if occurance.id in occurances:
                        occurances[occurance.id] += occurance.frequency
                    else:
                        occurances[occurance.id] = occurance.frequency

            for key1,value1 in occurances.items():
                for key2,value2 in occurances.items():
                    kernelMatrix[key1][key2] += value1*value2

        kernelMatrix = self.normalize(kernelMatrix)
        self.log("Kernel Matrix done")
        return kernelMatrix, leafIndex, kmerIDs, reverseIndex, directIndex

    def getLabels(self, data):
        output = [1 for i in range(9)]
        output.append(0)
        return output
        # return data[:,0].astype(int)

    def fristTraverseToPredict(self, data, classifier, K, maxMismatch):
        weights = dict()
        supportVectors = classifier.support_
        svData = []
        for i in supportVectors:
            svData.append(data[i])

        leaves, kmerIDs, reverseIndex = self.makeLeaves(svData, K, maxMismatch)

        for leaf in leaves:
            for node in leaf.nodeData:
                for occured in reverseIndex.get(kmerIDs.kmerIndex.get(node.kmerId)):
                    leaf.number += occured.frequency
            if leaf.number != 0:
                weights[leaf.word] = leaf.number
        return weights

    def firstTraverseFast(self, classifier, labels, leaves, reverseIndex, kmerIDs):
        weights = dict()
        SVAplhas = dict()
        for i in range(len(classifier.support_)):
            SVAplhas[classifier.support_[i]] = classifier.dual_coef_[0][i]
        SVs = set(classifier.support_)

        for leaf in leaves:
            for node in leaf.nodeData:
                for occured in reverseIndex.get(kmerIDs.kmerIndex.get(node.kmerId)):
                    if occured.id in SVs:
                        # SVAlphas is already negative for negative samples
                        leaf.number += occured.frequency*SVAplhas.get(occured.id)#*labels[occured.id]
            if leaf.number != 0:
                weights[leaf.word] = leaf.number

        rawWeights = copy.deepcopy(weights)

        for leaf in leaves:
            if leaf.word in weights:
                for node in leaf.nodeData:
                    neighbourKmer = kmerIDs.kmerIndex.get(node.kmerId)
                    if neighbourKmer in rawWeights:
                        weights[leaf.word] += rawWeights[neighbourKmer]


        return weights



    def firstTraverseFastFast(self, classifier, data, leafIndex, reverseIndex, kmerIDs, mismatch):
        weights = dict()
        SVAplhas = dict()
        for i in range(len(classifier.support_)):
            SVAplhas[classifier.support_[i]] = classifier.dual_coef_[0][i]
        SVs = set(classifier.support_)
        SVData = data[classifier.support_]
        supportKmers = []
        for d in SVData:
            kmers = Utils.getKMer(d[1], K)
            supportKmers.extend(kmers)

        for kmer in supportKmers:
            supportKmers.extend(Utils.neighbors(kmer,mismatch))

        supportKmers = set(supportKmers)

        for kmer in supportKmers:
            if kmer in leafIndex:
                leaf = leafIndex[kmer]
                for occurance in reverseIndex.get(kmer):
                    if occurance.id in SVs:
                        leaf.number += occurance.frequency * SVAplhas.get(occurance.id)  # *labels[occured.id]
                if leaf.number !=0:
                    weights[leaf.word] = leaf.number

        rawWeights = copy.deepcopy(weights)

        for kmer, weight in weights.items():
            leaf = leafIndex[kmer]
            for node in leaf.nodeData:
                neighbourKmer = kmerIDs.kmerIndex.get(node.kmerId)
                if neighbourKmer in rawWeights:
                    weights[leaf.word] += rawWeights[neighbourKmer]

        return weights


    def fastTraverse(self, classifier, leafIndex, reverseIndex, kmerIDs, directIndex):
        weights = dict()
        SVAplhas = dict()
        for i in range(len(classifier.support_)):
            SVAplhas[classifier.support_[i]] = classifier.dual_coef_[0][i]
        SVs = set(classifier.support_)

        relatedLeaves_withRedun = []
        for svRead, leafIdxs in directIndex.items():
            relatedLeaves_withRedun.extend(leafIdxs)

        relatedLeaves = set(relatedLeaves_withRedun)
        self.log("Related Leaves size " + str(len(relatedLeaves)))
        self.log("First Traverse")
        for leafId in relatedLeaves:
            leaf = leafIndex.get(leafId)
            for node in leaf.nodeData:
                for occured in reverseIndex.get(kmerIDs.kmerIndex.get(node.kmerId)):
                    if occured.id in SVs:
                        # SVAlphas is already negative for negative samples
                        leaf.number += occured.frequency*SVAplhas.get(occured.id)#*labels[occured.id]
            if leaf.number != 0:
                weights[leaf.word] = leaf.number

        rawWeights = copy.deepcopy(weights)
        self.log("Second Traverse")
        for leafId in relatedLeaves:
            leaf = leafIndex.get(leafId)
            if leaf.word in weights:
                for node in leaf.nodeData:
                    neighbourKmer = kmerIDs.kmerIndex.get(node.kmerId)
                    if neighbourKmer in rawWeights:
                        weights[leaf.word] += rawWeights[neighbourKmer]


        return weights
    

if __name__ == "__main__":
    data = Utils.loadSeqFile(path = "data/3.73.fold1.neg-train.seq")
    data = np.asarray(data)
    data = data[:10]

    compute = MismatchTree()

    labels = compute.getLabels(data)
    # kernelMat, leafIndex, kmerIDs, reverseIndex = makeKernel(data, K, M)
    kernelMat, leafIndex, kmerIDs, reverseIndex, directIndex = compute.makeKernel(data, K, M)
    print("kernelMatrix ", kernelMat)

    classifier = svm.SVC(kernel='precomputed', C=10)
    classifier.fit(kernelMat, labels)

    """
    first way:
    """
    # weights = fristTraverseToPredict(data, classifier, 3, 1)
    """
    second way:
    """
    # weights = firstTraverseFastFast(classifier, data,leafIndex, reverseIndex, kmerIDs, M)
    weights = compute.fastTraverse(classifier, leafIndex, reverseIndex, kmerIDs, directIndex)
    print("weights")
    for k,v in weights.items():
        print(k,v)

    np.save("weights-51.dat", weights, allow_pickle=True)
    np.save("kernelMatrix-51.dat", kernelMat, allow_pickle=True)

    # pred = classifier.predict(kernelMat)

    print("support vectors ", classifier.support_.shape, classifier.support_)
    # print("support labels ", labels[classifier.support_])
    print("alpha ", classifier.dual_coef_.shape, classifier.dual_coef_)

    # print("predictions ", pred)
    # print("labels ", labels)
    # print("error ", pred-labels)
    print(len(data))
    print(len(kernelMat))
    print(len(kernelMat[0]))
