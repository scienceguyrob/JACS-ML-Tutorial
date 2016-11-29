"""
This file is part of the JACS ML Tutorial.

JACS ML Tutorial is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

JACS ML Tutorial is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with JACS ML Tutorial.  If not, see <http://www.gnu.org/licenses/>.

File name:    SciKitLearnTest.py
Created:      October 30th, 2014
Author:       Rob Lyon

Contact:    rob@scienceguyrob.com or robert.lyon@postgrad.manchester.ac.uk
Web:        <http://www.scienceguyrob.com> or <http://www.cs.manchester.ac.uk>
            or <http://www.jb.man.ac.uk>

An example showing some of the capabilities of SciKit-Learn.

Designed to run on python 2.4 or later.

"""

# Classifier imports:
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA

# Evaluation imports:
from sklearn.metrics import confusion_matrix

# ROC Evaluation imports:
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# Cross validation imports:
from sklearn.cross_validation import StratifiedKFold

# Matlibplot imports:
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Custom imports:
import ARFF
import ClassifierStats

#*****************************
#
# CLASS DEFINITION
#
#*****************************

class SciKitLearnTest:
    """
    A simple example showing how to load ARFF data into SciKit-Learn,
    and use it for classification.

    """

    #*****************************
    #
    # MAIN METHOD AND ENTRY POINT.
    #
    #*****************************

    def main(self,argv=None):
        """
        Main entry point for the Application.

        """

        print "\n****************************"
        print "|                          |"
        print "|   SciKit-Learn Example   |"
        print "|                          |"
        print "|--------------------------|"
        print "| Version 1.0              |"
        print "| robert.lyon@cs.man.ac.uk |"
        print "***************************\n"

        # Please use only binary (two-class) data sets for this test script.
        # If using multi-class data sets you'll need to evaluate them slightly
        # differently, and luckily SciKit-Learn provides these tools for you.
        # If you just want to get up to speed, then convert your multi-class data
        # into binary form, by merging some of the classes into one. Remember in either
        # case you need to have data in the form:
        #
        # variable 1, variable 2, ... , variable n, class label
        #
        # where the class label must be an integer value, and the variables doubles, floats or integers.

        # Just alter the path to some data sets on your local machine.
        dataPath = "/Users/rob/Dropbox/Documents/SharedWorkspace/JACS_ML_Examples/data/magic.arff"

        # Create ARFF file object.
        arff = ARFF.ARFF()

        # Here I'm loading the test and training data from a single file.
        # data_X contains the data, and data_Y the correct class labels.
        data_X ,data_Y  = arff.read(dataPath)

        # In this example we have training and test data in a single file. so we need to
        # sample it in order to generate training and test data sets. This can be done as follows:
        #
        # train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(allData_X, allData_Y, test_size=0.4, random_state=0)
        #
        # This creates random test and train splits such that 40% of the data is used for testing. Note that
        # random_state is a random number used to seed a random sampler.
        #
        # It's equally valid to have test and training data sets sampled elsewhere. In which case you load
        # them separately as follows:
        #
        # trainPath    = "/Users/rob/.../trainData.arff"
        # testPath     = "/Users/rob/.../testData.arff"
        #
        # train_X , train_Y  = arff.read(trainPath)
        # test_X  , test_Y   = arff.read(testPath)
        #
        # Going back to our single data set example, we may not want to create simple random splits.
        # Often its better to create stratified splits, which preserve the ratio of each class
        # in a given sample. If you're not sure which type of sampling to use, Stratified sampling
        # is usually the better choice, since you know your training & test samples will have the same data
        # distribution.
        #
        # This is important since if you train on an unrepresentative distribution,
        # you may get strange performance in practice. This is because your classifier may 'overfit'
        # to the training distribution. For example: imagine you train a classifier upon data which is
        # 99% class one and 1% class two. Then it makes sense for the classifier to learn to always
        # predict class 1 since this strategy will always achieve the best accuracy. If the true
        # distribution is more 50:50 then clearly this classifier will make lots of mistakes!!!
        #
        # So here we build some stratified samples The data is split into 2 separate stratified folds.
        # The data are also shuffled so that we can be sure data ordering isn't a factor. Note the call to
        # StratifiedKFold simply chooses indices for the sample, and doesn't generate new data objects.
        skf = StratifiedKFold(data_Y, 2, shuffle=True,random_state=0)

        # The names of the classifiers being used...
        names = ["Linear SVM", "Decision Tree", "Random Forest", "Naive Bayes"]

        # Initializes and stores the actual classifiers to use.
        classifiers = [
                       SVC(kernel="linear", C=0.025,probability=True),
                       DecisionTreeClassifier(max_depth=5),
                       RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                       GaussianNB()
                       ]

        # Stores the results gathered below for comparison.
        aggregateResultsDict = {}

        # iterate over classifiers
        for name, classifier in zip(names, classifiers):

            # Some examples of using the classifiers provided below:

            print "Testing Classifier: " , name

            # Keeps count of the folds.
            fold = 0
            # Stores the results for each fold.
            results = []

            for train_index, test_index in skf:

                fold+=1

                print "\nRunning " , name , " on fold ", str(fold)
                train_X, test_X = data_X[train_index], data_X[test_index]
                train_Y, test_Y = data_Y[train_index], data_Y[test_index]

                # First train the classifier with call to fit.
                classifier.fit(train_X, train_Y)
                # Now obtain the classifiers 'score'
                accuracy = classifier.score(test_X, test_Y)
                print "\n",name," : ", accuracy # newline char at start simply for formatting.

                # But accuracy alone is insufficient to determine how well
                # a classifier performs on a test data set. So we really need
                # more metrics. An alternative to the above (and how we do things in ML) is to make
                # some predictions. Calling classifier.predict will return a set
                # of predicted labels which we can evaluate against test_Y (true labels).
                predictions = classifier.predict(test_X)

                # From the predictions we can compute a confusion matrix, which describes how
                # predicted labels were assigned. The matrix has the form:
                #
                #      Pred
                #        -     +
                # A    -----------
                # c  - | TN | FP |    TN = Negatives correctly receiving negative label.
                # t    |---------|    FN = Positives incorrectly receiving negative label.
                # u  + | FN | TP |    FP = Negatives incorrectly receiving positive label.
                # a    -----------    TP = Positives correctly receiving positive label.
                # l
                # ^
                # |
                # Predicted.
                confusionMatrix = confusion_matrix(test_Y,predictions)
                #print "Confusion matrix for ",name,":\n",confusionMatrix

                # Show confusion matrix in a separate window - uncomment if you'd
                # like to see it.
                #plt.matshow(confusionMatrix)
                #plt.title('Confusion matrix')
                #plt.colorbar()
                #plt.ylabel('True label')
                #plt.xlabel('Predicted label')
                #plt.show()

                # For this part of the evaluation, we assume the two classes used
                # in the test and training data sets represent positive (1) and negative (0).

                TrueNegatives  = confusionMatrix[0][0] # Negatives correctly receiving negative label.
                FalseNegatives = confusionMatrix[1][0] # Positives incorrectly receiving negative label.
                FalsePositives = confusionMatrix[0][1] # Negatives incorrectly receiving positive label.
                TruePositives  = confusionMatrix[1][1] # Positives correctly receiving positive label.

                # From this we obtain these key values from which we can
                # calculate classifier performance statistics.
                print "True Negatives  : " , TrueNegatives
                print "False Negatives : " , FalseNegatives
                print "False Positives : " , FalsePositives
                print "True Positives  : " , TruePositives, "\n\n"

                # From the values in the confusion matrix, we can calculate many performance metrics.
                # I've already written a class that allow you to automatically calculate
                # a whole bunch of these statistics. Here we construct this class object, then
                # print out the statistics calculated. Compare with the confusion matrix
                # values obtained above.
                classifierStats = ClassifierStats.ClassifierStats(confusionMatrix)
                print "More detailed statistics on ", name , " performance for fold " , fold , "."
                classifierStats.show()

                # But what if we want to produce a ROC curve? Well first we have to make sure
                # we are using a classifier that produces continuous output values, i.e probabilities.
                # Then the ROC curve can be computed. The ROC alters the threshold at which each instance
                # is classified. For example if the Threshold is 0.5, then we say everything receiving
                # a value below this is negative, and above is positive. Since we know the true class labels
                # we can draw a curve showing how many errors are made when altering this threshold, given predicted
                # probabilities and the the true class labels. By trying a variety of thresholds at small step
                # size, we produce a curve. See wikipedia for a detailed discussion of ROC curves. Even better,
                # check out this paper which describes them very well:
                # "An introduction to ROC analysis", Tom Fawcett, 2006 (https://ccrma.stanford.edu/workshops/mir2009/references/ROCintro.pdf)

                # Gets the probabilities. Note that we have to tell some classifiers explicitly
                # to output probabilities, e.g. SVC(kernel="linear", C=0.025,probability=True).
                probs = classifier.predict_proba(test_X)

                # Obtains the data for the curve.
                fpr, tpr, thresholds = roc_curve(test_Y, probs[:, 1])

                # Calculates the single area under the curve (AUC) metric. This is simply a
                # single metric which summarizes the ROC.
                roc_auc = auc(fpr, tpr)
                print "Area under the ROC curve : %f" % roc_auc

                # Store this value in the stats object for later use.
                classifierStats.setAUROC(roc_auc)

                plt.clf()
                plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.0])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic for ' + name + " on fold " + str(fold))
                plt.legend(loc="lower right")
                plt.show()

                # Now compute Precision-Recall and plot curve. This is very useful to get
                # a feel for how precise your classifier is. A very precise classifier with low positive
                # class recall is not helpful, likewise high recall with low precision is equally bad
                # (everything would be returned as a positive). So there is a trade-off to be had, and this
                # visualizes the trade-off.
                precision, recall, pr_thresholds =  precision_recall_curve(test_Y, probs[:, 1])

                # Calculates the single area under the precision-recall curve (AUPRC) metric.
                auprc = auc(recall,precision)
                print "Area under the PR curve : %f" % auprc

                # Store this value in the stats object for later use.
                classifierStats.setAUPRC(auprc)

                # Plot Precision-Recall curve
                plt.clf()
                plt.plot(recall, precision, label='Precision-Recall curve (area = %0.2f)' % auprc)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.ylim([0.0, 1.05])
                plt.xlim([0.0, 1.0])
                plt.title('Precision-Recall curve for ' + name + " on fold " + str(fold))
                plt.legend(loc="lower left")
                plt.show()

                # Now we have results per fold, but these need to be aggregated to get overall results
                # For now we store these in the results object.
                results.append(classifierStats)

                # Just some formatting to make it easier to visually separate fold results at the terminal.
                print "\n########################################\n"


            # Now outside of the loop over the folds, we can aggregate individual classifier performance.
            # This isn't the most efficient way to compute this (probably), but it's the easiest to follow
            # and improve upon.


            total_TP    = 0.0
            total_TN    = 0.0
            total_FP    = 0.0
            total_FN    = 0.0
            total_auroc = 0.0
            total_auprc = 0.0

            for cs in results:
                total_TP+= cs.getTP()
                total_TN+= cs.getTN()
                total_FP+= cs.getFP()
                total_FN+= cs.getFN()
                total_auroc+= cs.getAUROC()
                total_auprc+= cs.getAUPRC()

            tests = len(results)
            avg_TP    = total_TP    / tests
            avg_TN    = total_TN    / tests
            avg_FP    = total_FP    / tests
            avg_FN    = total_FN    / tests
            avg_auroc = total_auroc / tests
            avg_auprc = total_auprc / tests

            avgConfusionMatrix = [[avg_TN,avg_FN],[avg_FP,avg_TP]]
            aggregateStats = ClassifierStats.ClassifierStats(avgConfusionMatrix)
            aggregateStats.setAUROC(avg_auroc)
            aggregateStats.setAUPRC(avg_auprc)

            print "\nAggregate results for " , name , "\n"
            aggregateStats.show()

            aggregateResultsDict[name] = aggregateStats; # Add new entry

            # So now you have the data, what to do with it? Normally we would
            # Write it to a file for further analysis, for example significance
            # tests, ANOVA analysis etc.

            # Just some formatting to make it easier to visually separate results at the terminal.
            print "\n********************************************************************************\n"

        # Now we can use the aggregateResultsDict object to compare results
        # and determine which classifier is best for our data. For example:

        print "\nExample comparisons:\n"
        # NOTE: names[0]="Linear SVM" and names[1]="Decision Tree"
        if(aggregateResultsDict[names[0]].getGMean() < aggregateResultsDict[names[1]].getGMean()):

            diff = aggregateResultsDict[names[0]].getGMean()-aggregateResultsDict[names[1]].getGMean()
            print str(names[0]) , " achieved lower G-Mean than ", str(names[1]), ", diff: ", diff

        elif(aggregateResultsDict[names[0]].getGMean() > aggregateResultsDict[names[1]].getGMean()):

            diff = aggregateResultsDict[names[0]].getGMean()-aggregateResultsDict[names[1]].getGMean()
            print str(names[0]) , " achieved higher G-Mean than ", str(names[1]), ", diff: ", diff

        else:
            print str(names[0]) , " achieved the same G-Mean as ", str(names[1])

        # Or loop over the results....

        # Set initial best to compare to
        best = aggregateResultsDict[names[0]]
        name = names[0]
        for key in aggregateResultsDict:

            if(aggregateResultsDict[key].getGMean() > best.getGMean()):
                best = aggregateResultsDict[key]
                name = key

        print "Classifier that achieved the best G-Mean overall was" , name, " where G-Mean = ", str(best.getGMean())

        # Or for another metric....
        # Set initial best to compare to
        best = aggregateResultsDict[names[0]]
        name = names[0]
        for key in aggregateResultsDict:

            if(aggregateResultsDict[key].getFP() > best.getFP()):
                best = aggregateResultsDict[key]
                name = key

        print "Classifier that produced the most false positives was" , name, " where FP = ", str(best.getFP())

        print "\nDone."

    #***************************************************************************************************

if __name__ == '__main__':
    SciKitLearnTest().main()
