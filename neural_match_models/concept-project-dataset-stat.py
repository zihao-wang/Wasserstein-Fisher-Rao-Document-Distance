import os
import pickle
import json

datasetDir = "..\\data\\concept-project"
conceptsFile = os.path.join(datasetDir, 'concepts.txt')
projectsFile = os.path.join(datasetDir, 'projects.txt')
annotateFile = os.path.join(datasetDir, 'annotations.txt')
recordPickle = os.path.join(datasetDir, 'record.pickle')
dictionaryFile = os.path.join(datasetDir, 'dict.json')
# task 1: extract records

projectContents = []
with open(projectsFile, 'rt', encoding='utf8') as pf:
    for l in pf.readlines():
        content = l.strip().split('\t||\t')[-1].split()[1:]
        projectContents.append(content)

conceptContents = []
with open(conceptsFile, 'rt', encoding='utf8') as cf:
    for l in cf.readlines():
        content = l.strip().split(':')[-1].split()
        conceptContents.append(content)

annotates = []
with open(annotateFile, 'rt', encoding='utf8') as af:
    for l in af.readlines():
        label = int(l.strip())
        annotates.append(int(label))

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
with open(recordPickle, 'wb') as f:
    pickle.dump(records, f)
with open(dictionaryFile, 'wt') as f:
    json.dump(dictionary, f)
