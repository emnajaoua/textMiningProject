from preprocessing import _getVerbSubjPairs
from preprocessing import writePDinList
from preprocessing import parseSentence
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import itertools
import os


def getTokenizeListFRomData(text, stem=True):
    TokenizeList = []
    reviewTextList, summaryList = writePDinList()  # list1 corresponds to reviewText, list2 corresponds to summary
    for reviewText in reviewTextList:
        d, T = parseSentence(reviewText)
        TokenizeList.append(_getVerbSubjPairs(d, T)[2])
    return TokenizeList

def preprocessPhrases(Tokenizer):
    parsedSentence = [word for word in Tokenizer if
                      word not in ["'s", '%', '?', '``', ',', '--', '.', '/', '.', "''", '$', '&']]
    for word in parsedSentence:
        if (re.findall(r'\d', word)):
            parsedSentence.remove(word)
    parsedSentence = [word for word in parsedSentence if word.lower() not in stopwords.words('english')]
    return parsedSentence

def processTroFiDataset():
    verbList = []
    filtered_words_non_literal = []
    filtered_words_literal = []
    subjectClustersNonLiteral = []
    subjectClustersLiteral = []
    objectClustersLiteral = []
    objectClustersNonLiteral = []
    with open('./testTroFi.txt') as f:
        content = f.readlines()
    content = [x.strip() for x in content] #To remove \n from lines
    x = content
    i = 0
    print('len(content)', len(content))
    while i != len(content):
        if re.findall(r'\*\*\*(\w){3,}\*\*\*', x[i]):
            verb = x[i]
            verbList.append(verb[3:])
            i += 1
        elif (re.findall(r'\*(nonliteral cluster)\*', x[i])):
            l = x[i+1]
            while (i != len(content) and (not (re.findall(r'\*(literal cluster)\*', l)))):
                nonLiteral = re.findall(r'([N]([\t]))(.*)', x[i])
                literal = re.findall(r'([L]([\t]))(.*)', x[i])
                if nonLiteral:
                    str = nonLiteral[0][2]
                    dependency_parser, Tokenizer = parseSentence(str)
                    subjectClustersNonLiteral += _getVerbSubjPairs(dependency_parser, Tokenizer)[1]
                    objectClustersNonLiteral+= _getVerbSubjPairs(dependency_parser, Tokenizer)[2]
                    parsedSentence = preprocessPhrases(Tokenizer)
                    filtered_words_non_literal.append(parsedSentence)
                elif literal:
                    str = literal[0][2]
                    dependency_parser, Tokenizer = parseSentence(str)
                    subjectClustersLiteral += _getVerbSubjPairs(dependency_parser, Tokenizer)[1]
                    objectClustersLiteral += _getVerbSubjPairs(dependency_parser, Tokenizer)[2]
                    parsedSentence = preprocessPhrases(Tokenizer)
                    filtered_words_literal.append(parsedSentence)
                i += 1
        elif (re.findall(r'\*(literal cluster)\*', x[i]) and (not (re.findall(r'\*(nonliteral cluster)\*', x[i])))):
            l = x[i+1]
            while (i != len(content) and not(re.findall(r'(\*){4,}',l))):
                nonLiteral = re.findall(r'([N]([\t]))(.*)', x[i])
                literal = re.findall(r'([L]([\t]))(.*)', x[i])
                if nonLiteral:
                    dependency_parser, Tokenizer = parseSentence(str)
                    subjectClustersNonLiteral += _getVerbSubjPairs(dependency_parser, Tokenizer)[1]
                    objectClustersNonLiteral += _getVerbSubjPairs(dependency_parser, Tokenizer)[2]
                    parsedSentence = preprocessPhrases(Tokenizer)
                    filtered_words_non_literal.append(parsedSentence)
                elif literal:
                    str=literal[0][2]
                    dependency_parser, Tokenizer = parseSentence(str)
                    subjectClustersLiteral += _getVerbSubjPairs(dependency_parser, Tokenizer)[1]
                    objectClustersLiteral += _getVerbSubjPairs(dependency_parser, Tokenizer)[2]
                    parsedSentence = preprocessPhrases(Tokenizer)
                    filtered_words_literal.append(parsedSentence)
                i +=1
    return verbList, filtered_words_non_literal, filtered_words_literal, subjectClustersNonLiteral, objectClustersNonLiteral, subjectClustersLiteral, objectClustersLiteral

# verbList, filtered_words_non_literal, filtered_words_literal, subjectClustersNonLiteral, objectClustersNonLiteral, subjectClustersLiteral, objectClustersLiteral = processTroFiDataset()
# print('verbList', verbList)
# print('filtered_words_non_literal', filtered_words_non_literal)
# print('filtered_words_literal', filtered_words_literal)
# print('subjectClustersNonLiteral', subjectClustersNonLiteral)
# print('subjectClustersLiteral', subjectClustersLiteral)
# print('objectClustersNonLiteral', objectClustersNonLiteral)
# print('objectClustersLiteral', objectClustersLiteral)


def buildModel(phrases):
    eval_path = './evaluation/similarity'
    # define training data
    # phrases = filtered_words_non_literal
    # train model
    model = Word2Vec(phrases, min_count=1)
    # summarize the loaded model
    # print(model)
    # summarize vocabulary
    # words = list(model.wv.vocab)
    # print(words)
    # access vector for one word
    # print(model['absorb'])
    # save model
    model.save('model.bin')
    # load model
    new_model = Word2Vec.load('model.bin')
    return new_model
# verbList = ['absorb']
# filtered_words_non_literal = [['But', 'short-term', 'absorb', 'lot', 'top', 'management', 'energy', 'attention', 'says', 'Philippe', 'Haspeslagh', 'business', 'professor', 'European', 'management', 'school', 'Insead', 'Paris'], ['The', 'Monitor', 'losses', 'absorbed', 'church', 'working', 'fund', 'reportedly', 'declined', 'past', 'two', 'years', '200', 'million', '280', 'million', 'largely', 'stock-market', 'crash']]
# subjectClustersNonLiteral = ['it', 'Haspeslagh', 'which']
# objectClustersNonLiteral = ['lot']
# model = buildModel(filtered_words_non_literal)
# verbListLabels = ["absorb"]
# sentenceTest = ["Haspeslagh", "absorb", "a", "lot"]

def calculateSimilarityToModel(model, sentenceTest, subjectClusters, objectClusters):
    d,t = parseSentence(sentenceTest)
    preprcessedToken = preprocessPhrases(t)
    average_similarity_subject_object = 0
    average_similarity = 0
    averageSubjects = 0
    averageObjects = 0
    for subject in subjectClusters:
        i = 0
        while i != len(preprcessedToken):
            try:
                if (model.n_similarity([subject], [preprcessedToken[i]])):
                    average_similarity += model.n_similarity([subject], [preprcessedToken[i]])
            except KeyError:
                print(preprcessedToken[i], "or",subject, "not in vocabulary")
            i += 1
    averageSubjects += average_similarity / i
    average_similarity = 0
    for object in objectClusters:
        i = 0
        while i != len(preprcessedToken):
            try:
                if (model.n_similarity([object], [preprcessedToken[i]])):
                    average_similarity += model.n_similarity([object], [preprcessedToken[i]])
            except KeyError:
                print(t[i], "or", object, "not in vocabulary")
            i += 1
    averageObjects += average_similarity/i
    verbList, subjectList, objectList = _getVerbSubjPairs(d,t)
    for i, j in itertools.product(range(len(objectList)), range(len(subjectList))):
        try:
            if (model.n_similarity([objectList[i]], [subjectList[j]])):
                average_similarity_subject_object += model.n_similarity([objectList[i]], subjectList[j])
        except KeyError:
            print(objectList[i], "or", subjectList[j], "not in vocabulary")
    average = (averageObjects + averageSubjects + average_similarity_subject_object) / 3
    return average

def labelPhrase(phrase, averageLiteral, averageNonLiteral):
    if averageLiteral > averageNonLiteral:
        predictedLabel = "L"
        print(phrase, "is not metaphor")
    else:
        print(phrase, "is metaphor")
        predictedLabel = "N"
    return predictedLabel

if __name__ == "__main__":
    PhrasesLiteral = ["Haspeslagh absorb a lot", "The management of this country is good"]
    PhrasesNonLiteral = ["The management of this country is bad"]
    PhrasesLiteralDict = dict([(p, "L") for p in PhrasesLiteral])
    PhrasesNonLiteralDict = dict([(p, "N") for p in PhrasesNonLiteral])
    Phrases = {**PhrasesLiteralDict, **PhrasesNonLiteralDict}
    correctLiteralsInLiteralCluster = 0
    correctLiterals = 0
    for testPhrase, label in Phrases.items():
        verbList, filtered_words_non_literal, filtered_words_literal, subjectClustersNonLiteral, objectClustersNonLiteral, subjectClustersLiteral, objectClustersLiteral = processTroFiDataset()
        modelNonLiteral = buildModel(phrases=filtered_words_non_literal)
        averageNonLiteral = calculateSimilarityToModel(modelNonLiteral, testPhrase, subjectClustersNonLiteral, objectClustersNonLiteral)
        modelLiteral = buildModel(phrases=filtered_words_literal)
        averageLiteral = calculateSimilarityToModel(modelLiteral, testPhrase, subjectClustersNonLiteral, objectClustersNonLiteral)
        predictedLabel = labelPhrase(testPhrase, averageLiteral, averageNonLiteral)
        if ((predictedLabel == label) & ( testPhrase in PhrasesLiteralDict)) :
            correctLiteralsInLiteralCluster += 1
        if (predictedLabel == label):
            correctLiterals +=1
    recall = correctLiteralsInLiteralCluster / correctLiterals
    precision = correctLiteralsInLiteralCluster / len(PhrasesLiteralDict)
    print('precision', precision)
    print('recall', recall)




# list1 = ["subject1", "subject2"]
# list2 = ["object1", "object2", "object3"]
# for i,j in itertools.product(range(len(list1)), range(len(list2))):
#     print (i,j)
#     print (list1[i], list2[j])