import os
import pickle
import json

datasetDir = "..\\data\\mrpc"
trainFile = os.path.join(datasetDir, 'msr_paraphrase_test.txt')
testFile = os.path.join(datasetDir, 'msr_paraphrase_train.txt')
recordPickle = os.path.join(datasetDir, 'record.pickle')
dictionaryFile = os.path.join(datasetDir, 'dict.json')

# task 1: extract records

projectContents = []
with open(trainFile, 'rt', encoding='utf8') as pf:
    pf.readline()
    for l in pf.readlines():
        content = l.strip().split('\t')[-1].split()
        projectContents.append(content)

with open(testFile, 'rt', encoding='utf8') as pf:
    pf.readline()
    for l in pf.readlines():
        content = l.strip().split('\t')[-1].split()
        projectContents.append(content)

print(len(projectContents))


conceptContents = []
with open(trainFile, 'rt', encoding='utf8') as pf:
    pf.readline()
    for l in pf.readlines():
        content = l.strip().split('\t')[-2].split()
        conceptContents.append(content)

with open(testFile, 'rt', encoding='utf8') as pf:
    pf.readline()
    for l in pf.readlines():
        content = l.strip().split('\t')[-2].split()
        conceptContents.append(content)

print(len(conceptContents))

annotates = []
with open(trainFile, 'rt', encoding='utf8') as pf:
    pf.readline()
    for l in pf.readlines():
        content = l.strip().split('\t')[0]
        annotates.append(int(content))

with open(testFile, 'rt', encoding='utf8') as pf:
    pf.readline()
    for l in pf.readlines():
        content = l.strip().split('\t')[0]
        annotates.append(int(content))

# task 2: count the words and make the dictionary

dictionary = {'NA': 0}
maxProjectLength = 0
for pc in projectContents:
    maxProjectLength = max(maxProjectLength, len(pc))
    for w in pc:
        if w not in dictionary:
            dictionary[w] = len(dictionary)
print(maxProjectLength)

maxConceptLength = 0
for cc in conceptContents:
    maxConceptLength = max(maxConceptLength, len(cc))
    for w in cc:
        if w not in dictionary:
            dictionary[w] = len(dictionary)
print(maxConceptLength)

print(len(dictionary))

# task 3: make mappings

projectWordID = [
    [dictionary[w] for w in pc] for pc in projectContents
]

conceptWordID = [
    [dictionary[w] for w in cc] for cc in conceptContents
]

# task 4: combine records
# record format (project content, concepts content, annotation)

records = [(pc, cc, a) for pc, cc, a in zip(projectWordID, conceptWordID, annotates)]
print(len(records))
with open(recordPickle, 'wb') as f:
    pickle.dump(records, f)
with open(dictionaryFile, 'wt') as f:
    json.dump(dictionary, f)
