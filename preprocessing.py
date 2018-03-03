import pandas as pd
import gzip
import re
from pycorenlp import StanfordCoreNLP
from stanfordcorenlp import StanfordCoreNLP
import nltk.data

def __init(self):
    self.reviewTextList = []
    self.summaryList = []
    self.TokenizeList = []

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g: yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

df = getDF('reviews_Automotive_5.json.gz')
# print(df.head())
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def writePDinList():
    reviewTextList = []
    summaryList = []
    for index, row in df.iterrows():
        summaryList.append(row['summary'])
        for sentence in tokenizer.tokenize(row['reviewText']):
            reviewTextList.append(sentence)
    return reviewTextList, summaryList

print('test1', writePDinList()[0][0:8])

def parseSentence(sentence):
    nlp = StanfordCoreNLP( r'/root/PycharmProjects/textmining/textmining/textMiningProject/stanford-corenlp-full-2018-01-31')

    # sentence = 'Guangdong University of Foreign Studies is located in Guangzhou.'
    Tokenize = nlp.word_tokenize(sentence)
    # part_of_speech = nlp.pos_tag(sentence)
    # named_entities = nlp.ner(sentence)
    # constituency_parser = nlp.parse(sentence)

    dependency_parser = nlp.dependency_parse(sentence)
    # regex_pattern = r"\(NN (\w+)\)"
    # const_parse = constituency_parser
    # print('const parse', const_parse)
    # NN_list = re.findall(r"\(NN (\w+)\)", const_parse)
    # NNS_list = re.findall(r"\(NNS (\w+)\)", const_parse)
    #
    # noun_list = NN_list + NNS_list
    nlp.close() # Do not forget to close! The backend server will consume a lot memory.
    return dependency_parser, Tokenize

print('dependency parser', parseSentence(writePDinList()[0][1])[0])
print('Tokenizer' , parseSentence(writePDinList()[0][1])[1])

def _getVerbSubjPairs(dependency_parser, Tokenizer):
    subjectVerbList = []
    verbObjList = []
    TokenizeList = []
    for element in dependency_parser:
        if element[0] == 'nsubj':
            verb = element[1]
            subject = element[2]
            subjectVerbList.append((Tokenizer[verb - 1], Tokenizer[subject - 1]))
            TokenizeList.append(Tokenizer[subject - 1])
        elif element[0] == 'dobj':
            verb_index = element[1]
            obj_index = element[2]
            verbObjList.append((Tokenizer[verb_index - 1], Tokenizer[obj_index - 1]))
            TokenizeList.append(Tokenizer[obj_index - 1])
    return subjectVerbList, verbObjList, TokenizeList

def getTokenizeListFRomData(df):
    TokenizeList = []
    reviewTextList, summaryList = writePDinList()  # list1 corresponds to reviewText, list2 corresponds to summary
    for reviewText in reviewTextList:
        TokenizeList.append(_getVerbSubjPairs(parseSentence(reviewText))[2])
    return TokenizeList

    # extract all the pairs (subject, verb) and (verb, object) from the dependency parser
            # subjectVerbList represent the pairs verb-subject
            # verbObjList represent the pairs verb-object

            # compute the log likelihood between the pairs (statistical method)


            # Distance = 1/ loglikelihood => the longer the conceptual distance is, and thus the higher the probability of a metaphoric expression is

            # if things doesn't workout : use add-one smoothing method or include the LL between the hyponyms and hypernyms of the verbs

            # Clustering approach

            # use the TroFi dataset because it is annotated (Literal, non literal)

            # use the k-means algorithms :
            # inputs: Let X= {x1, x2, …, xn} be a set of subject/object points in training corpus.
            # By means of the WordNet resource, we use the JC similarity2 measure to compute the similarities between every two points in X;

# dependency_parser = parseSentence("My dog also likes eating sausage")[0]
# subjectVerbList = []
# verbObjList = []
# for element in dependency_parser:
#     if element[0] == 'nsubj':
#         verb = element[1]
#         subject = element[2]
#         subjectVerbList.append((Tokenizer[verb - 1], Tokenizer[subject - 1]))
#     elif element[0] == 'dobj':
#         verb_index = element[1]
#         obj_index = element[2]
#         verbObjList.append((Tokenizer[verb_index - 1], Tokenizer[obj_index - 1]))
#
# print('subjectVerb', subjectVerbList)
# print('verbObj', verbObjList)

# def dependency_parse(sentence):
#     """ Accepts a sentence and returns its dependency parse as a list-of-lists
#     Also returns the list of nouns in the sentence
#     """
#     parse_output = parseSentence(sentence)
#
#     # List of nouns
#     const_parse = parse_output[0]
#     print(const_parse)
#     regex_pattern = r"\(NN (\w+)\)"
#     NN_list = re.findall(r"\(NN (\w+)\)", const_parse)
#     NNS_list = re.findall(r"\(NNS (\w+)\)", const_parse)
#
#     noun_list = NN_list + NNS_list
#
#     # Dependency parse
#     dep_parse = parse_output[1].split("\n")
#
#     print("---")
#     dependency_parse=[]
#     for i in dep_parse:
#         if len(i.strip()) > 0 and i.strip()[0] != "(":
#             line=i.strip()
#             dependency_parse.append(filter(lambda x:x.isalpha(),re.findall(r"[\w']+", line)))
#
#     return dependency_parse, noun_list

#parsing
# for s in summaryList:
#     print ('sentence', s)
#     dependency_parser, noun_list = parseSentence(s)
#     print('dependency parser', dependency_parser)
#     print('noun_list', noun_list)