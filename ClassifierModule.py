import string
import math
import string
import re
from collections import Counter
from itertools import chain

class Classifier(object):

    vocab = set() #all individual words of the data set
    num_of_messages = {} #all the messages of the data set

    log_class_priors = {} #prior log probability of each class
    word_counts = {}  #how many times each word appears in each class
    word_log_probabilities = {} #the log probabilities of each word for each class
    class_of_message = {}
    

    #this data will be written to text files
    data_word_model = {} #probability of each word appearing in each label
    data_model = {} #each line with it's respectice label
    data_results = {} #each line with its predicted label and actual label

    def clean(self, s):
        translator = str.maketrans("", "", string.punctuation)
        return s.translate(translator)
    
    def tokenize(self, text):
        str1 = ' '.join(text)
        text = self.clean(str1).lower()
        return re.split("\W+", text)

    #returns how many times a word appears overall
    def get_word_counts(self, docs):
        word_counts = {}
        for word in docs:
            str1 = ''.join(word)
            word_counts[str1] = word_counts.get(str1, 0.0) + 1.0
        return word_counts

    #returns data structure of how many times each word appears in each class
    def get_word_frequencies(self):
        return self.get_word_counts

    #----------------------------TASK 1--------------------------------
    def train_nb(self, docs, labels):

        num_of_lines = len(docs)    
        self.num_of_messages['pos'] = sum(1 for label in labels if label == "pos")
        self.num_of_messages['neg'] = sum(1 for label in labels if label == "neg")

        #get log probabilities of main classes (neg and pos)
        self.log_class_priors['pos'] = math.log(self.num_of_messages['pos'] / num_of_lines)
        self.log_class_priors['neg'] = math.log(self.num_of_messages['neg'] / num_of_lines)

        #hold word counts for each class
        self.word_counts['pos'] = {}
        self.word_counts['neg'] = {}

        #get how many times each word appears for each label pos and neg
        for doc, label in zip(docs, labels):
            c = 'pos' if label == 'pos' else 'neg'
            counts = self.get_word_counts(self.tokenize(doc))

            message = ' '.join(doc)
            self.class_of_message[message] = label

            for word, count in counts.items():
                if word not in self.vocab:
                    self.vocab.add(word)
                if word not in self.word_counts[c]:
                    self.word_counts[c][word] = 0.0

                self.word_counts[c][word] += count
        return self.word_counts
    
    #----------------TASK 2-----------------------
    #need to find probability of each word appearing in a label
    #word_counts gives how many times each word appears in a label
    def score_doc_label(self, doc, label):
        result = {}
        
        self.word_log_probabilities['pos'] = {}
        self.word_log_probabilities['neg'] = {}

        for word in self.vocab:
            log_prob_word_given_pos = math.log( (self.word_counts['pos'].get(word, 0.0) + 0.5) / (self.num_of_messages['pos']) )
            log_prob_word_given_neg = math.log( (self.word_counts['neg'].get(word, 0.0) + 0.5) / (self.num_of_messages['neg']) )
            self.word_log_probabilities['pos'][word] = log_prob_word_given_pos
            self.word_log_probabilities['neg'][word] = log_prob_word_given_neg

            self.data_word_model[word] = ("POS probability: ", self.word_log_probabilities['pos'][word], "NEG probability: ", self.word_log_probabilities['neg'][word])

        return self.word_log_probabilities

    #classify a new document
    def classify_nb(self, doc, label):
        result = []
        word_counts_prob = self.score_doc_label(doc, label)
        for line in doc:
            counts = self.get_word_counts(self.tokenize(i) for i in line) 
            pos_score = 0
            neg_score = 0
            for word, _ in counts.items():
                if word not in self.vocab: continue
                log_prob_word_given_pos = word_counts_prob['pos'][word]
                log_prob_word_given_neg = word_counts_prob['neg'][word]

                pos_score += log_prob_word_given_pos
                neg_score += log_prob_word_given_neg
                self.data_results[word] = ()

            pos_score += self.log_class_priors['pos']
            neg_score += self.log_class_priors['neg']

            if(pos_score > neg_score):
                result.append("pos")
            else:
                result.append("neg")
        return result

    #-----------------------TASK 3--------------------------------
    #return a list of what class each document belongs to 
    #docs is a list of strings
    def classify_documents(self, docs):
        result = []
        temp = "pos"
        #for each line in the document, apply the bayes algorithm and append the result 
        for line in docs:
            counts = self.get_word_counts(self.tokenize(i) for i in line)
            pos_score = 0
            neg_score = 0
            for word, _ in counts.items():
                if word not in self.vocab: continue

                #get the log probability of a word appearing in each class and add smoothing of 0.5
                log_prob_word_given_pos = math.log( (self.word_counts['pos'].get(word, 0.0) + 0.5) / (self.num_of_messages['pos']) )
                log_prob_word_given_neg = math.log( (self.word_counts['neg'].get(word, 0.0) + 0.5) / (self.num_of_messages['neg']) )

                pos_score += log_prob_word_given_pos
                neg_score += log_prob_word_given_neg
                
                self.data_model[word] = ("POS score: ", pos_score, "NEG score: ", neg_score)

            pos_score += self.log_class_priors['pos']
            neg_score += self.log_class_priors['neg']

            if pos_score > neg_score:
                result.append("pos")
                temp = "pos"
            else:
                result.append("neg")
                temp = "neg"
            string = ' '.join(line)
            self.data_results[string] = ("POS probability: ", pos_score, ", NEG probability: ", neg_score, ", Classified as: ", temp, ", Actual class:", self.class_of_message[string])
        return result
        

            
