ALPHABET = ['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V' ]

def getKMer(string, k):
    """
        Returns k-mer from string by sliding window
    """
    b = [string[i:i + k] for i in range(len(string) - (k - 1))]
    return b

def loadSeqFile(path):
    """
        Load seq file and parses entries like below into [[label, feature]] list
            >0_d1hcw__ 9.8.1.1.1 23-residue designed metal-free peptide based on the zinc finger domains [synthetic]
            YTVPSTFSRSDELAKLLRLHAG
    """
    data = []

    def record(label, feature):
        if len(label.strip()) > 0 and len(feature.strip())>0:
            classlabel = -1 if label=='0' else 1
            data.append([classlabel, feature])

    with open(path) as file:
        lines = file.readlines()
        classlabel = ''
        feature = ''
        for line in lines:
            if ">" == line[0]:
                record(classlabel, feature)
                classlabel = line[1]
                feature = ''
            else:
                feature += line.replace("\n", '')
        record(classlabel, feature)

    return data[1:]

def hammingDist(str_1, str_2):
    mismatch_num = 0
    for idx in range(len(str_1)):
        if str_1[idx] != str_2[idx]:
            mismatch_num += 1
    return mismatch_num

def neighbors(kmer, d):
    neighborhood = set([])
    if d == 0:
        return [kmer]
    if len(kmer) == 1:
        return ALPHABET
    suffix_neighbors = neighbors(kmer[1:], d)
    for string in suffix_neighbors:
        if hammingDist(string, kmer[1:]) < d:
            for sym in ALPHABET:
                neighborhood.add(sym+string)
        else:
            neighborhood.add(kmer[0]+string)
    return list(neighborhood)
    
    
# print(len(ALPHABET))
# print(len(neighbors("AGD",1)))  

    
    
    

