import json, lxml, nltk, pickle, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from glob import glob
from lxml import etree
from math import ceil
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from sklearn.model_selection import StratifiedShuffleSplit

#np.random.seed(13)

punkt_param = PunktParameters()
# List of common abbreviations present in the source XML files
abbreviations = ['c', 'cent', 'dr', 'eliz', 'etc', 'hon', 'mr', 'no', 'nos', 'vol']
punkt_param.abbrev_types = set(abbreviations)
# Tokenize with the abbreviations passed as parameters
sent_tokenizer = PunktSentenceTokenizer(punkt_param)

class DataGenerator:

    def __init__(self, filedir, test_size, batch_size):
        """filedir    : The path to the XML corpus directory;
           test_size  : The size of the validation partition, e.g. 0.3;
           batch_size : The number of instances per batch."""
        self.filedir = filedir
        debates = self.process_xml(filedir)
        debates, w2i = self.process_debates(debates)
        self.debates = debates
        self.w2i = w2i
        self.test_size = test_size
        X_train, Y_train, X_test, Y_test = self.generate_splits(self.debates, test_size)
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.batch_size = batch_size
        train_batches, train_lengths = self.generate_batches(self.X_train, self.Y_train, batch_size)
        self.train_batches = train_batches
        self.train_lengths = train_lengths
        test_batches, test_lengths = self.generate_batches(self.X_test, self.Y_test, batch_size)
        self.test_batches = test_batches
        self.test_lengths = test_lengths

    @staticmethod
    def process_xml(path):
        """Parse XML files from a directory specified as path."""
        #filenames, debates = glob(path + '\*.xml'), []
        filenames, debates = glob(path + '//*.xml'), []
        for filename in filenames:
            debate = []
            # Iterative parsing for optimised memory usage
            # (_ stands for event, etree.iterparse() is the context)
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

    @staticmethod
    def process_debates(debates):
        """Perform text cleaning, sentence tokenisation, and word tokeni-
           sation as well as debate tagging, vocabulary building, and sub-
           sequent integer encoding of words and boundary tags."""

        def tag_debate(debate):
            """Mark boundaries between sentences in each debate as either [SAME] or
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
            # Remove any remaining empty strings, sentences, and speeches,
            # and add special tokens to mark sentence boundaries
            debate = [[['<s>'] + [w for w in s if w] + ['</s>'] \
                       for s in speech if s] for speech in debate if speech]
            # Tag the entire debate text within each speech and
            # between speeches as [SAME] and [CHANGE] respectively
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
        return encoded_debates, w2i

    @staticmethod
    def generate_splits(debates, test_size): # (self, test_size)
        """Randomize and preserve the distribution of labels across the data
           by shuffling instances, and splitting into train and test subsets;
           test_size indicates the size of a test partition, e.g.: 0.2."""
        # Pack the data and labels into the respective arrays
        X = np.array([(triple[0], triple[1]) for triple in debates]) #self.debates
        Y = np.array([triple[2] for triple in debates])              #self.debates
        # Initialize the cross validator with a single split
        # and the specified size of the test partition
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=13)
        # indices = sss.split(X, Y)
        # train_indices, test_indices = indices[0][0], indices[0][1]
        # Generate the mask with indices and perform the stratified splitting
        for train_index, test_index in sss.split(X, Y):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
        return X_train, Y_train, X_test, Y_test
    
    @staticmethod
    def generate_batches(X_array, Y_array, batch_size): #(self, X_array, Y_array, batch_size)
        """Divide the arrays of inputs and targets into
           batches of the specified size, e.g.: 32."""
        # Separate training instances for two RNNs
        X_array_left = np.array([X_array[i][0] for i in range(len(X_array))])
        X_array_right = np.array([X_array[i][1] for i in range(len(X_array))])
        # Get lengths of left and right sentences
        X_length_left = np.array([len(x) for x in X_array_left])
        X_length_right = np.array([len(x) for x in X_array_right])
        # Generate a shuffled array of indices
        indices = np.random.permutation(len(X_array))
        # Re-arrange the input, length, and label arrays
        # with respect to the shuffled order of indices
        X_array_left = X_array_left[indices]
        X_array_right = X_array_right[indices]
        Y_array = Y_array[indices]
        X_length_left = X_length_left[indices]
        X_length_right = X_length_right[indices]
        # Find the number of batches (batch_index) that can be produced
        # for the selected length of each batch (batch_size) and divide
        # the array of indices into the same number of partitions
        batch_indices = [indices[batch_index * batch_size : (batch_index + 1) * batch_size] \
                     for batch_index in range(ceil(len(indices) / batch_size))]
        batches = [( X_array_left[arr], X_array_right[arr], Y_array[arr] ) for arr in batch_indices]
        lengths = [( X_length_left[arr], X_length_right[arr] ) for arr in batch_indices]
        batches = [( nn.utils.rnn.pad_sequence([torch.LongTensor(xl) for xl in batch[0]], batch_first=True), \
                     nn.utils.rnn.pad_sequence([torch.LongTensor(xr) for xr in batch[1]], batch_first=True), \
                     torch.FloatTensor(batch[2]) ) for batch in batches]
        lengths = [( torch.LongTensor(lb[0]), torch.LongTensor(lb[1]) ) for lb in lengths] 
        return batches, lengths

    def statistics(self):
        """Print out the essential statistics about the processed data."""
        print('Total instances: {}'.format(len(self.debates)), end='\n')
        print('Number of unique tokens: {}'.format(len(self.w2i)), end='\n')
        print('Test set size: {}%'.format(self.test_size*100), end='\n')
        print('Number of training instances: {}'.format(len(self.X_train)), end='\n')
        print('Number of test instances: {}'.format(len(self.X_test)), end='\n')
        Y = [triple[2] for triple in self.debates]
        class_proportion = [Counter(Y)[label] / len(Y) for label in set(Y)]
        class_proportion_train = [Counter(self.Y_train)[label] / len(self.Y_train) for label in set(self.Y_train)]
        class_proportion_test = [Counter(self.Y_test)[label] / len(self.Y_test) for label in set(self.Y_test)]
        print('Class distribution (entire corpus): [SAME] : {:5.2f}%, [CHANGE] : {:5.2f}%'.format(
            class_proportion[0]*100, class_proportion[1]*100), end='\n')
        print('Class distribution (training set): [SAME] : {:5.2f}%, [CHANGE] : {:5.2f}%'.format(
            class_proportion_train[0]*100, class_proportion_train[1]*100), end='\n')
        print('Class distribution (test set): [SAME] : {:5.2f}%, [CHANGE] : {:5.2f}%'.format(
            class_proportion_test[0]*100, class_proportion_test[1]*100), end='\n')
        print('Batch size: {} instances per batch'.format(self.batch_size), end='\n')
        print('Number of training batches: {}'.format(len(self.train_batches)), end='\n')
        print('Number of validation batches: {}'.format(len(self.test_batches)), end='\n')

#if __name__ == '__main__':
#    data_generator = DataGenerator('...//scrapedxml//debates', test_size=0.2, batch_size=32)
#    data_generator.statistics()

#Enter your filepath to the XML files to extraxt debates
#UNIX-style path: '...\scrapedxml\debates'
#data_generator = DataGenerator('...', test_size=test_size, batch_size=batch_size)
