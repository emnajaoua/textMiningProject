## TO DO according to this paper Identifying English Verb Metaphors Using Statistical and Clustering Approaches

# stanford parser to analyze the dependency relations between the verb and its subject/object in a sentence
from preprocessing import _getVerbSubjPairs
from preprocessing import writePDinList
from preprocessing import parseSentence
import string
import collections

from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint

# def setClusters(w):
#     synonyms = []
#     for syn in wordnet.synsets(w):
#         for lemma in syn.lemmas():
#             synonyms.append(lemma.name())


def getTokenizeListFRomData(text, stem=True):
    TokenizeList = []
    reviewTextList, summaryList = writePDinList()  # list1 corresponds to reviewText, list2 corresponds to summary
    for reviewText in reviewTextList:
        TokenizeList.append(_getVerbSubjPairs(parseSentence(reviewText))[2])
    return TokenizeList


# def process_text(text, stem=True):
#     """ Tokenize text and stem words removing punctuation """
#     text = text.translate(None, string.punctuation)
#     tokens = word_tokenize(text)
#
#     if stem:
#         stemmer = PorterStemmer()
#         tokens = [stemmer.stem(t) for t in tokens]
#
#     return tokens

def cluster_texts(texts, clusters=3):
    """ Transform texts to Tf-Idf coordinates and cluster texts using K-Means """
    vectorizer = TfidfVectorizer(tokenizer=getTokenizeListFRomData,
                                 stop_words=stopwords.words('english'),
                                 max_df=0.5,
                                 min_df=0.1,
                                 lowercase=True)

    tfidf_model = vectorizer.fit_transform(texts)
    km_model = KMeans(n_clusters=clusters)
    km_model.fit(tfidf_model)

    clustering = collections.defaultdict(list)

    for idx, label in enumerate(km_model.labels_):
        clustering[label].append(idx)

    return clustering


if __name__ == "__main__":
    text = writePDinList()[0]
    clusters = cluster_texts(text, 7)
    pprint(dict(clusters))

    # after forming the clusters according to the k-means algorithm, we compute average similarity between clusters
    # of the training corpus and the words in the base metaphoric resources which is TroFi
