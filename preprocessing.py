import pandas as pd
import gzip
from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')

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
print('test', writePDinList()[1])
text = (
  'Pusheen and Smitha walked along the beach. '
  'Pusheen wanted to surf, but fell off the surfboard.')
output = nlp.annotate(text, properties={
  'annotators': 'tokenize,ssplit,pos,depparse,parse',
  'outputFormat': 'json'
  })
print(output['sentences'][0]['parse'])

