import os
import json
from textblob.classifiers import NaiveBayesClassifier
from textblob.classifiers import DecisionTreeClassifier
from textblob.classifiers import NLTKClassifier
import time
import nltk.classify
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
import re

import nltk
nltk.download('punkt')

#Defining Classes to work with nltk's classifyl interface
#This interface allows strings to be used as labels

#This one contains Naive bayes Classifier
class SVMClassifier(NLTKClassifier):


    nltk_class = nltk.classify.SklearnClassifier(LinearSVC())

#This one is for Rocchio Classifier
class RocchioClassifier(NLTKClassifier):


    nltk_class = nltk.classify.SklearnClassifier(NearestCentroid())

#This one is for K Nearest Neighbors
class NNClassifier(NLTKClassifier):


    nltk_class = nltk.classify.SklearnClassifier(KNeighborsClassifier())


#This returns a list which contrains addresses of all the text files from a given directory
def list_files_from_directory(directory):

    ret_val = [] #list which stores
    for file in os.listdir(directory):
        if file.endswith(".txt"): #Text files only
            ret_val.append(str(directory) + "/" + str(file))
    #print("The first 5 directories of list are: \n",ret_val[:5])
    return ret_val


#This reads a file from a given path and returns it as a list
def read_file(path):

    f = open(path, "r")
    read = f.readlines()
    ret_val = []
    #while reading we want to ignore lines that starts with # and read other
    for line in read:
        if line.startswith("#"):
            pass
        else:
            ret_val.append(line)
    #print("The first 5 lines of list are: \n",ret_val[:5])
    return ret_val

#Now we want to split the line whenever there is a tab
def process_line(line):
    """Returns sentence category and sentence in given line"""

    if "\t" in line: #tab spacing condition
        splits = line.split("\t")

        #first word/column of every line is a category
        s_category = splits[0]

        #Second column is of the sentence and we are converting it to lower case
        sentence = splits[1].lower()

        #removing stopwords
        for sw in stopwords:
            sentence = sentence.replace(sw, "")
        #Compile a regular expression pattern into a regular expression object which can be used for different operations
        pattern = re.compile("[^\w']")

        #Removing space and plus symbol, if any, from the left side of string
        sentence = pattern.sub(' ', sentence)
        sentence = re.sub(' +', ' ', sentence)
        return s_category, sentence
    else:
        splits = line.split(" ") #if spliting is by space, first word of the only column is category, rest is sentence
        s_category = splits[0]
        sentence = line[len(s_category)+1:].lower()
        for sw in stopwords:
            sentence = sentence.replace(sw, "")
        pattern = re.compile("[^\w']")
        sentence = pattern.sub(' ', sentence)
        sentence = re.sub(' +', ' ', sentence)
        return s_category, sentence

#Writes training data from given folder into formatted JSON file, which stores all files as text
def create_json_file(input_folder, destination_file):

    tr_folder = list_files_from_directory(input_folder) #Getting paths for all txt files
    all_json = []
    for file in tr_folder:
        lines = read_file(file)
        for line in lines:
            c, s = process_line(line) #Splitting into category and sentence
            if s.endswith('\n'):
                s = s[:-1] #Removes the last character
            json_data = {
                'text': s,
                'label': c
            }
            all_json.append(json_data)

    with open(destination_file, "w") as outfile:
        json.dump(all_json, outfile)

#Performaing same steps for test data
def prepare_test_data(input_folder):
    """Maps each sentence to it's category"""

    test_folder = list_files_from_directory(input_folder) #Getting paths for all txt files
    t_sentences = []
    t_categories = []
    for file in test_folder:
        lines = read_file(file)
        for line in lines:
            c, s = process_line(line) #splitting into catergory and sentence
            if s.endswith('\n'): #removing last character if it's \n
                s = s[:-1]
            t_sentences.append(s)
            t_categories.append(c)
    return t_categories, t_sentences

# main

# loading stopwords
input_stopwords = read_file("word_lists/stopwords.txt")
stopwords = []
for word in input_stopwords:
    if word.endswith('\n'): #removing last character if it's \n
        word = word[:-1]
        stopwords.append(word)


# prepare training and test data
create_json_file("training_set", "training.json")
categories, sentences = prepare_test_data("test_set")

# Bayes Classifier
print("\n\nNaive Bayes Classifier's Training started")
start_time = time.time() #start time count

with open('training.json', 'r') as training: #training classifier and taking json file as output
    nbc = NaiveBayesClassifier(training, format="json")
stop_time = time.time()

print("Naive Bayes Classifier's Training completed")
elapsed = round((stop_time - start_time),2)
print("Training time (in seconds): " + str(elapsed),"s")
print("Testing Naive Bayes Classifier")
correct = 0
start_time = time.time()
for i in range(0, len(sentences)): #running till the length of the string
    category = str(nbc.classify(sentences[i])).lower() #Prediction
    expected = str(categories[i]).lower() #ground truth
    if category == expected: #Checking
        correct += 1 #if correct then increasing the counter
stop_time = time.time() #stopping time
elapsed = round((stop_time - start_time),2) #calculating time and rounding off
print("Number of tests: " + str(len(sentences)))
print("Correct tests: " + str(correct))
accuracy = round((correct / len(sentences))*100,2) #calculating and rounding accuracy
print("Naive Bayes Classifier accuracy on Test Data: " + str(accuracy),"%")
print("Testing time (in seconds): " + str(elapsed),"s")

#Below three have the same replication instead of just classifier


# Rocchio
print("\n \nTraining Rocchio Classifier...")
start_nbc = time.time()
with open('training.json', 'r') as training:
    nbc = RocchioClassifier(training, format="json")
stop_time = time.time()
print("Training Rocchio Classifier completed...")
elapsed = round((stop_time - start_time),2)
print("Training time (in seconds): " + str(elapsed),"s")
print("Testing Rocchio Classifier...")
correct = 0
start_nbc = time.time()
for i in range(0, len(sentences)):
    category = str(nbc.classify(sentences[i])).lower()
    expected = str(categories[i]).lower()
    if category == expected:
        correct += 1
stop_time = time.time()
elapsed = round((stop_time - start_time),2)
print("Number of tests: " + str(len(sentences)))
print("Correct tests: " + str(correct))
accuracy = correct / len(sentences)
print("Rocchio Classifier accuracy: " + str(accuracy),"%")
print("Testing time (in seconds): " + str(elapsed),"s")

# Knearest

print("\n \nTraining Knearest Classifier...")
start_time = time.time()
with open('training.json', 'r') as training:
    nbc = NNClassifier(training, format="json")
stop_time = time.time()
print("Training Knearest Classifier completed...")
elapsed = round((stop_time - start_time),2)
print("Training time (in seconds): " + str(elapsed),"s")
print("Testing Knearest Classifier...")
correct = 0
start_time = time.time()
for i in range(0, len(sentences)):
    category = str(nbc.classify(sentences[i])).lower()
    expected = str(categories[i]).lower()
    if category == expected:
        correct += 1
stop_time = time.time()
elapsed = round((stop_time - start_time),2)
print("Number of tests: " + str(len(sentences)))
print("Correct tests: " + str(correct))
accuracy = correct / len(sentences)
print("Knearest Classifier accuracy: " + str(accuracy),"%")
print("Testing time (in seconds): " + str(elapsed),"s")
