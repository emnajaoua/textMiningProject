import pandas as pd
import gzip
import re
from pycorenlp import StanfordCoreNLP
from stanfordcorenlp import StanfordCoreNLP


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
def writePDinList():
    reviewTextList = []
    summaryList = []
    for index, row in df.iterrows():
        reviewTextList.append(row['reviewText'])
        summaryList.append(row['summary'])
    return reviewTextList, summaryList

list1, list2 = writePDinList()
# print('test', writePDinList()[1])
# ##get constituency parser
# nlp = StanfordCoreNLP('http://localhost:9000')
#
# text = (
#   'Pusheen wanted to surf, but fell off the surfboard.')
# output = nlp.annotate(text, properties={
#   'annotators': 'tokenize,ssplit,pos,depparse,parse',
#   'outputFormat': 'json'
#   })
# print(output['sentences'][0]['parse'])
def parseSentence(sentence):
    nlp = StanfordCoreNLP( r'/root/PycharmProjects/textmining/textmining/textMiningProject/stanford-corenlp-full-2018-01-31')

    # sentence = 'Guangdong University of Foreign Studies is located in Guangzhou.'
    print('Tokenize:', nlp.word_tokenize(sentence))
    print('Part of Speech:', nlp.pos_tag(sentence))
    print('Named Entities:', nlp.ner(sentence))
    print('Constituency Parsing:', nlp.parse(sentence))
    print('Dependency Parsing:', nlp.dependency_parse(sentence))
    nlp.close() # Do not forget to close! The backend server will consume a lot memery.
    return nlp.dependency_parse(sentence)

def dependency_parse(sentence):
    """ Accepts a sentence and returns its dependency parse as a list-of-lists
    Also returns the list of nouns in the sentence
    """
    parse_output = parseSentence(sentence)

    # List of nouns
    const_parse = parse_output[0]
    print(const_parse)
    regex_pattern = r"\(NN (\w+)\)"
    NN_list = re.findall(r"\(NN (\w+)\)", const_parse)
    NNS_list = re.findall(r"\(NNS (\w+)\)", const_parse)

    noun_list = NN_list + NNS_list

    # Dependency parse
    dep_parse = parse_output[1].split("\n")

    print("---")
    dependency_parse=[]
    for i in dep_parse:
        if len(i.strip()) > 0 and i.strip()[0] != "(":
            line=i.strip()
            dependency_parse.append(filter(lambda x:x.isalpha(),re.findall(r"[\w']+", line)))

    return dependency_parse, noun_list
