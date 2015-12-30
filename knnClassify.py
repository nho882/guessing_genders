from math import sqrt
import string
import sys
import os

__author__='Nancy Ho and Aoi Yamamoto'

# Feature Functions for Step 1

def length(name):
    """feature function with length of name"""
    return [('length', len(name))]

def vowels(name):
    """feature function with number of vowels in name"""
    def isVowel(c):
        return c in 'aeiouy'
    return [('vowels', len(filter(isVowel, name)))]

# Write the four additional feature functions below
def hardCons(name):
    def isHardCon(l):
        #returns the letter if it is a hard consonant
        return l in 'ptkbdg'
    #returns the len of the list of hard consonants
    return ['hard', len(filter(isHardCon, name))]

#puts the alphabet into the alphabet variable
alphabet = string.lowercase
            
def countLetters(name):
    def count(s):
        #returns the count of a letter
        return ('count-' + s, name.count(s))
    #returns the count for each letter
    return map(count, alphabet)

def lastLetter(name):
    def last(s):
        #returns true if the last letter is the current letter
        return ('last-'+ s, int(s == name[-1]))
    #goes through the list of alphabets
    return map(last, alphabet)

def firstLetter(name):
    def first(s):
        #same as lastLetter 
        return ('first-' + s, int(s == name[0]))
    return map(first, alphabet)
    
# Feature Combination for Step 2
def combine(featureFunc1, featureFunc2):
    """returns a function that, when applied to a name, 
    returns a feature vector
    combining the feature vectors from featureFunc1 and featureFunc2"""
    # fill in
    def com(name):
        #combines the functions into a list
        return featureFunc1(name) + featureFunc2(name)
    return com

def combineMany(listOfFeatureFuncs):
    """returns a function that, when applied to a name, returns a feature vector
    combining the feature vectors from featureFunc1 and featureFunc2"""
    # fill in
    def com(name):
        #puts the functions into one list 
        return reduce(combine, listOfFeatureFuncs)(name)
    return com
    
# Classification for Step 3

def euclideanDistanceFrom(v):
    """returns a function that takes a vector and computes the distance between that vector and v"""
    def euclideanDistanceHelper(w):
        """distance between vectors v and w"""
        def squareDiff(i):
            """square of difference between elements in ith position of vectors v and w"""
            return (v[i][1] - w[i][1])**2
        return sqrt(sum(map(squareDiff, range(len(v)))))
    return euclideanDistanceHelper

# 3a
def buildTrainingVectors(featurefunc, trainfile):
    """read names from trainfile, return a list of tuples,
    where each tuple in the list corresponds to a name.
    Each tuple is of the form (genderlabel, featurevector),
    where genderlabel is the gender of the name,
    and featurevector is the feature vector of the name under featurefunc.
    """
    # fill in
    #opens the file to read the lines
    inputFile = open(trainfile, 'r')
    #use map function to go over each line to modify data
    contents = map(lambda x: x.strip().split(), inputFile)
    #close the file
    inputFile.close()
    #returns data in the proper format
    return map(lambda x: (x[1], featurefunc(x[0])), contents)
    

# 3b
def labelsSortedByDistance(testvector, trainingVectors):
    """create a list of (genderlabel, distance) tuples from trainingVectors
    (trainingVectors is a list of (gender, featvec) tuples, 
    of the type returned by buildTrainingVectors).
    The distance is the Euclidean distance between featvec and testvector.
    Sort this list of tuples by distance"""
    # fill in
    #helper functions to get items
    def getItem1(item):
        return item[0]
    
    def getItem2(item):
        return item[1]
    
    #makes the distance function to compare the other vectors to the test vector
    distanceFromTest = euclideanDistanceFrom(testvector)
    #featureVecs gets the actual feature vectors
    featureVecs = map(getItem2, trainingVectors)
    #listOfGenders gets the genders from the training vectors
    listOfGenders = map(getItem1, trainingVectors)
    
    #puts together the list of tuples with gender, and then uses the distance function
    #to compare each with the test vector
    finalList = zip(listOfGenders, map(distanceFromTest, featureVecs))
    #sorts the list according to distance
    finalList2 = sorted(finalList, key=getItem2)
    return finalList2

# 3c
def predictGender(featurefunc, testname, trainingVectors, k):
    """find the k nearest training vectors to testname's
    feature vectors.
    return the most common label among those nearest training vectors."""
    # fill in
    #creates the vector for the testname
    testVector = featurefunc(testname)
    #makes the sorted labels according to the testname and training vectors
    sortedLabels = labelsSortedByDistance(testVector, trainingVectors)
    #gets the list up until and including k items
    searchLabels = sortedLabels[:k]
    #returns first item
    def getFirst(item):
        return item[0]
    #gets the list of genders 
    gender = map(getFirst, searchLabels)
    
    #predicts gender according to which is more
    if gender.count('male') > gender.count('female'):
        return 'male'
    else:
        return 'female'
    
    
def computeAccuracy(featurefunc, testfile, trainingVectors, k):
    """Go through each name in testfile, predict its gender label,
    and print the predictions for each example as instructed.
    Return the accuracy (the proportion of examples for which 
    the labels are correctly predicted)."""
    # fill in
    #opens file, reads it, and closes file
    inputFile = open(testfile, 'r')
    contents = map(lambda x: x.strip().split(), inputFile)
    inputFile.close()
    
    count = 0
    
    #for each name, predict the gender, and up the count if correct
    for name in contents:
        guessGender = predictGender(featurefunc, name[0], trainingVectors, k)
        if name[1] == guessGender:
            answer = 'CORRECT'
            count += 1
        else:
            answer = 'WRONG'
        print name[0], name[1], guessGender, answer
    #returns the percentage correct
    return float(count)/len(contents)

# Step 4
def main():
    """Runs the classification pipeline with command line arguments"""
    # fill in
    try:
        #converts k into an int
        k = int(sys.argv[1])
        #goes through the list of featurefuncs to make them from string to function
        featurefuncsList = map(lambda x: eval(x), sys.argv[2:])
        #combine them into one list
        combinedFuncs = combineMany(featurefuncsList)
        #create the training Vectors
        trainedVectors = buildTrainingVectors(combinedFuncs, 'train.txt')
        #compare the training to the test data
        accuracy = computeAccuracy(combinedFuncs, 'test.txt', trainedVectors, k)
        print accuracy
    except:
        print 'Arguments to this program: k featurefunc1 featurefunc2... where k is an integer,and each featurefuncn is a valid feature function'
        

if __name__=='__main__':  # invoke main() when program is run
    main()
