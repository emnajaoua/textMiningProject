""" GetDependencyParse.py - Contains functions to deal with output of Stanford's dependency parser
"""

import re
from preprocessing import parse

def dependency_parse(sentence):
    """ Accepts a sentence and returns its dependency parse as a list-of-lists
    Also returns the list of nouns in the sentence
    """

    parse_output = parse()[1]

    # List of nouns
    const_parse = parse_output[0]
    print (const_parse)
    regex_pattern = r"\(NN (\w+)\)"
    NN_list = re.findall(r"\(NN (\w+)\)", const_parse)
    NNS_list = re.findall(r"\(NNS (\w+)\)", const_parse)

    noun_list = NN_list + NNS_list

    # Dependency parse
    dep_parse = parse_output[1].split("\n")

    print ("---")
    dependency_parse=[]
    for i in dep_parse:
        if len(i.strip()) > 0 and i.strip()[0] != "(":
            line=i.strip()
            dependency_parse.append(filter(lambda x:x.isalpha(),re.findall(r"[\w']+", line)))

    return dependency_parse, noun_list

sentence = parse()[1][0]
print (dependency_parse(sentence))