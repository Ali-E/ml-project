import numpy as np
import Utils
from sklearn.metrics import accuracy_score


kernelMatrix = np.load("kernelMatrix-51.dat.npy")
weights = np.load("weights-51.dat.npy")

weights = weights.item()


data = Utils.loadSeqFile(path = "data/3.73.fold1.neg-train.seq")
# data.extend(Utils.loadSeqFile(path = "/Users/akash/Downloads/fisher-scop-data/3.73/3.73.1.2/3.73.1.1:d1pot__.pos-test.seq"))
# data.extend(Utils.loadSeqFile(path = "/Users/akash/Downloads/fisher-scop-data/1.1/1.1.fold0.neg-train.seq"))

data = np.asarray(data)

labels = data[:,0].astype(int)

def predict(data):
    kmers = Utils.getKMer(data[1], 5)
    f = 0
    for kmer in kmers:
        if kmer in weights:
            f += weights.get(kmer)
    return f


predictions = []
for d in data:
    predictions.append(predict(d))

for i in range(len(predictions)):
    print(np.sign(predictions[i]), predictions[i], labels[i])

predictions = np.asarray(predictions)
print(accuracy_score(labels, np.sign(predictions)))
