import json, lxml, nltk, pickle, random
import numpy as np
import pandas as pd
from collections import Counter
from glob import glob
from lxml import etree
from math import ceil
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from sklearn.model_selection import StratifiedShuffleSplit

np.random.seed(13)

punkt_param = PunktParameters()
# List of common abbreviations present in the source XML files
abbreviations = ['c', 'cent', 'dr', 'eliz', 'etc', 'hon', 'mr', 'no', 'nos', 'vol']
punkt_param.abbrev_types = set(abbreviations)
# Tokenize with the abbreviations passed as parameters
sent_tokenizer = PunktSentenceTokenizer(punkt_param)

def process_xml(path):
    """Parse XML files from a directory specified as path."""
    filenames, debates = glob(path + '//*.xml'), []
    for filename in filenames:
        debate = []
        # Iterative parsing for optimised memory usage (_ stands for event, etree.iterparse() is the context)
        for _, element in etree.iterparse(filename, tag='speech'):
            # Delete all sub-elements with the specified tags but preserve
            # and merge their content into the parent "speech" element.
            etree.strip_tags(element, *['phrase', 'i', 'table', 'tbody', 'tr', 'td'])
            debate.append(' '.join(' '.join(etree.tostring(
                node, encoding='unicode', method='text').split()) for node in element))
            element.clear()
        debates.append(debate)
    # Return only non-empty debates
    return list(filter(bool, debates)) # Same as (debate for debate in debates if debate), or list(filter(None, debates))

def tag_debate(debate):
    """Mark boundaries between sentences in a debate as either [SAME] or
       [CHANGE], depending on whether they are uttered by the same speaker
       or different speakers."""
    tagged_debate = []
    for i in range(len(debate)-1):
        for j in range(len(debate[i])-1):
            if len(debate[i]) > 1:
                tagged_debate.append( (debate[i][j], debate[i][j+1], '[SAME]') )
        tagged_debate.append( (debate[i][-1], debate[i+1][0], '[CHANGE]') )
    if len(debate[-1]) > 1:
        for i in range(len(debate[-1])-1):
            tagged_debate.append( (debate[-1][i], debate[-1][i+1], '[SAME]') )
    return tagged_debate

# Same, but as a list comprehension
# debate_tagged = [ [(debate[i][j], debate[i][j+1], '[SAME]')
#                   for j in range(len(debate[i])-1) if len(debate[i]) > 1] + \
#                  [(debate[i][-1], debate[i+1][0], '[CHANGE]')]
#                  for i in range(len(debate)-1)] + \
#                  [(debate[-1][i], debate[-1][i+1], '[SAME]')
#                   for i in range(len(debate[-1])-1) if len(debate[-1]) > 1]

def process_debate(debates):
    """Perform text cleaning, sentence tokenisation, and word tokeni-
       sation as well as debate tagging, vocabulary building, and sub-
       sequent integer encoding of words and boundary tags."""
    tagged_debates, vocabulary = [], []
    for debate in debates:
        # Minor text cleaning to unglue mdashes, hyphens, ellipses, and apostrophes from words
        debate = (s.replace("—", " — ") for s in debate)
        debate = (s.replace("-", " - ") for s in debate)
        debate = (s.replace("…", " … ") for s in debate)
        debate = (s.replace("'", "' ") if s.startswith("'") else s for s in debate)
        # Tokenize each speech into separate sentences
        debate = (sent_tokenizer.tokenize(speech.lower()) for speech in debate)
        # Tokenize each sentence in the speech into words
        debate = ((nltk.word_tokenize(s.lower()) for s in speech) for speech in debate)
        # Filter out remaining rogue apostrophes and periods
        debate = [[[w.replace("'", '') if w.startswith("'") and w != "'s" else w \
                for w in s] for s in speech] for speech in debate]
        debate = [[[w.replace(".", '') if w.endswith(".") and w != "." else w \
                for w in s] for s in speech] for speech in debate]
        # Remove any remaining empty strings, sentences, and speeches
        debate = [[[w for w in s if w] for s in speech if s] for speech in debate if speech]
        # Tag the entire debate text within each speech and between speeches as [SAME] and [CHANGE] respectively
        tagged_debate = tag_debate(debate)
        # Retrieve all unique words from a single debate, to append to the vocabulary
        debate_words = list(set(w for speech in debate for s in speech for w in s))
        tagged_debates.append(tagged_debate), vocabulary.append(debate_words)
    vocabulary = ['<pad>'] + list(set(w for debate_words in vocabulary for w in debate_words))
    w2i = dict(zip(vocabulary, range(len(vocabulary))))
    # Encode features and labels with integers
    encoded_debates = [([w2i[w] for w in triple[0]], \
                        [w2i[w] for w in triple[1]], \
                        {'[SAME]' : 0, '[CHANGE]' : 1}[triple[2]]) \
                       for tagged_debate in tagged_debates \
                       for triple in tagged_debate]
    # encoded_debates[2]
    # ([6826, 3466, 1312], [7090, 3417, 3838, 7000, 1797, 3912], 1)
    # Wrap the vectorized features and labels in the DataFrame
    #df = pd.DataFrame(encoded_debates, columns=['left', 'right', 'label'])
    return encoded_debates

def generate_splits(data, n):
    """Randomize and preserve the distribution of labels across the data
       by shuffling instances, and splitting into train and test subsets;
       n denotes the size of a test partition, e.g.: 0.2."""
    # Pack the data and labels into the respective arrays
    X = np.array([(triple[0], triple[1]) for triple in data])
    Y = np.array([triple[2] for triple in data])
    # Find the proportion between the classes
    # class_proportion = [Counter(Y)[label] / len(Y) for label in set(Y)]
    # Initialize the cross validator with a single split
    # and the specified size of the test partition
    sss = StratifiedShuffleSplit(n_splits=1, test_size=n, random_state=13)
    # indices = sss.split(X, Y)
    # train_indices, test_indices = indices[0][0], indices[0][1]
    # Generate the mask with indices and perform the stratified splitting
    for train_index, test_index in sss.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
    return X_train, Y_train, X_test, Y_test

def generate_batches(X_array, Y_array, batch_size):
    """Divide the arrays of inputs and targets into batches of the
       specified size, e.g.: 32."""
    indices = np.random.permutation(len(X_array))

    X_train = X_train[indices,:]
    Y_train = Y_train[indices]

    batch_indices = [indices[batch_index * batch_size : (batch_index + 1) * batch_size] for batch_index in range(ceil(len(indices) / batch_size))]
    batches = [( X_array[arr], Y_array[arr] ) for arr in batch_indices]
    return batches

#Enter your filepath to the XML files to extraxt debates
#debates = process_xml('...\scrapedxml\debates')
#Clean, integer encode, and label each pair of sentences
#encoded_debates = process_debate(debates)
#Split into training and test sets
#X_train, Y_train, X_test, Y_test = generate_splits(encoded_debates, 0.2)
#Generate batches from the data
#train_batches = generate_batches(X_train, Y_train, 32)
