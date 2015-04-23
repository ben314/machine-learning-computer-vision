
# coding: utf-8

# In[1]:

"""
EXERCISE 1: Dataset Preparation
"""

import struct
import numpy as np
import matplotlib.pyplot as pyplot

# Wherever the training and test data are stored
filepath = "C:/Users/Ben/Dropbox/2- Master TUM/5. Semester - SS15/Machine Learning in Computer Vision/Exercises/exercise_1/code/"

filename_training_images = "train-images.idx3-ubyte"
filename_training_labels = "train-labels.idx1-ubyte"
filename_test_images = "t10k-images.idx3-ubyte"
filename_test_labels = "t10k-labels.idx1-ubyte"

def plot_image( pixel_vector, cols=28, rows=28 ):
    
    image_array = np.zeros( (cols, rows, 3) )
    
    i = 0
    
    for x in xrange( cols ):
        for y in xrange( rows ):
            image_array[x,y] = np.array([255-pixel_vector[i]] * 3, dtype="uint8")
            i += 1
   
    pyplot.imshow( image_array )
    
def read_image_file( filename_images, buffer_size=1024 ):

    with open(filepath + filename_images, 'rb') as file_images:

        print "Reading images from " + filename_images + " ..."

        # START Reading header
        magic_number_byte = file_images.read(4)
        magic_number = struct.unpack(">i",magic_number_byte)[0]

        number_of_images_byte = file_images.read(4)
        number_of_images = struct.unpack(">i",number_of_images_byte)[0]

        number_of_rows_byte = file_images.read(4)
        number_of_rows = struct.unpack(">i",number_of_rows_byte)[0]

        number_of_columns_byte = file_images.read(4)
        number_of_columns = struct.unpack(">i",number_of_columns_byte)[0]
        # END Reading header
        
        pixels_per_image = number_of_rows * number_of_columns
        
        # data_set_array will contain row vectors representing images
        data_set_array = np.zeros( (number_of_images, pixels_per_image), dtype="uint8" )
        
        for image_idx in xrange(number_of_images) :
        # for image_idx in xrange(2) :        
            
            pixel_bytes = file_images.read(pixels_per_image)

            for pixel_idx in xrange(pixels_per_image):
                pixel = struct.unpack(">B",pixel_bytes[pixel_idx])[0]
                data_set_array[image_idx, pixel_idx] = pixel

    print "Finished reading images from " + filename_images
    
    return data_set_array

def read_label_file( filename_labels ):
    
    with open(filepath + filename_labels, 'rb') as file_labels:

        print "Reading labels from " + filename_labels + " ..."

        # START Reading header
        magic_number_byte = file_labels.read(4)
        magic_number = struct.unpack(">i",magic_number_byte)[0]
        
        number_of_items_byte = file_labels.read(4)
        number_of_items = struct.unpack(">i",number_of_items_byte)[0]
        # END Reading header
        
        label_array = np.zeros( number_of_items, dtype="uint8" )
        
        for label_idx in xrange( number_of_items ):
            
            label_byte = file_labels.read(1)
            label = struct.unpack(">B",label_byte)[0]
            
            label_array[label_idx] = label
        
    print "Finished reading labels from " + filename_labels
    
    return label_array


# In[1]:




# In[2]:

""" Load label files """
array_training_labels = read_label_file( filename_training_labels )
array_test_labels =     read_label_file( filename_test_labels )


# In[3]:

""" Load image files """
array_training_images = read_image_file( filename_training_images )
array_test_images =     read_image_file( filename_test_images )


# In[4]:

data_index = 59999
print "Label:", array_training_labels[data_index]
plot_image( array_training_images[data_index] )


# In[5]:

data_index = 9999
print "Label:", array_test_labels[data_index]
plot_image( array_test_images[data_index] )


# In[6]:

"""
EXERCISE 2: Support Vector Machines
"""

from sklearn.externals import joblib
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
import time

def try_different_gammas( number_of_samples = 1000 ):

    best_estimate = ( NaN, 100.0 )

    for g in xrange( 11 ):

        gamma = g / 10000000.0

        # Use radial base function as kernel (gaussian in this case) 
        clf = svm.SVC( kernel='rbf', gamma=gamma )

        if number_of_samples:
            clf.fit( array_training_images[:number_of_samples], array_training_labels[:number_of_samples] )  
        else:
            clf.fit( array_training_images, array_training_labels )  

        predicted_labels = clf.predict( array_test_images )

        f1 = f1_score( array_test_labels, predicted_labels )
        precision = precision_score( array_test_labels, predicted_labels )
        recall = recall_score( array_test_labels, predicted_labels )
        
        error_rate = 100.0 * (1.0 - accuracy_score( array_test_labels, predicted_labels ))
        
        if( error_rate<best_estimate[1] ):
            best_estimate = ( gamma, error_rate )

        # roc = roc_auc_score( array_test_labels, predicted_labels )

        print "gamma = %0.10f" % gamma, "f1 = %0.3f" % f1 , "error rate (%%) = %0.3f" % error_rate

    return best_estimate

def classifier_filename( gamma ):
    
    return 'classifiers/svm_gamma_' + str(gamma) + '.pkl'

def learn_svm( number_of_samples=0, gamma=0.0000004, **kwargs ):

    print "Started learning SVM classifier..."

    # Use radial base function as kernel (gaussian in this case) 
    clf = svm.SVC( kernel='rbf', gamma=gamma, **kwargs )

    if number_of_samples:
        clf.fit( array_training_images[:number_of_samples], array_training_labels[:number_of_samples] )  
    else:
        clf.fit( array_training_images, array_training_labels )  

    joblib.dump( clf, filepath + classifier_filename(gamma) )
    
    print "Finished learning SVM classifier. Result saved in " + classifier_filename(gamma)
    
def evaluate_svm( gamma=0.0000004 ):
    
    print "Started evaluating SVM classifier..."
    
    clf = joblib.load( filepath + classifier_filename(gamma) )
    predicted_labels = clf.predict( array_test_images )
    
    error_rate = 100.0 * (1.0 - accuracy_score( array_test_labels, predicted_labels ))
    evaluation_report = classification_report( array_test_labels, predicted_labels )
    
    print "Finished evaluating SVM classifier.\n"
    
    print "SVM classifier evaluation"
    # print "gamma = %0.10f" % gamma, "f1 = %0.6f" % f1 , "precision = %0.6f" % precision, "recall = %0.6f" % recall
    
    print "gamma = %0.10f" % gamma + "\nerror rate (%%) = %0.6f" % error_rate
    print evaluation_report


# In[7]:

""" 
Exercise 2.3-2.5
Learn models using 1000 samples of the training set.
try_different_gammas() prints the tried gamma values and f1, precision and recall scores
"""
# try_different_gammas( number_of_samples=1000 )


# In[8]:

"""
Exercise 2.6 
learn_svm() learns a model using the whole training set.
The trained model is stored in a file by learn() and loaded by evaluate_svm().
The learning process took more than 20 minutes, so
it is wrapped in a function learn_full_training_set()
such that it can be called if desired
"""
gamma = 0.0000004
def learn_full_training_set():

    times = []

    t1 = time.clock()
    learn_svm( gamma=gamma )
    t2 = time.clock()
    print "Time elapsed: %i\n" % int(t2-t1)
    
#learn_full_training_set()


# In[9]:

"""
evaluate_svm() prints the gamma value, error rate and classification report
"""
# evaluate_svm( gamma=gamma )


# In[12]:

"""
EXERCISE 3: Decision Trees and Random Forests
"""
from sklearn import tree
from sklearn.externals.six import StringIO  

"""
Exercise 3.1(a)
"""
def train_decision_tree_3_1_a():
    
    print "Started learning default Decision Tree classifier.\n"
    
    clf = tree.DecisionTreeClassifier()
    clf.fit( array_training_images, array_training_labels )
    
    print "Finished learning default Decision Tree classifier.\n"
    
    print "Started evaluating default Decision Tree classifier.\n"
    
    predicted_labels = clf.predict( array_test_images )
    
    error_rate = 100.0 * (1.0 - accuracy_score( array_test_labels, predicted_labels ))
    evaluation_report = classification_report( array_test_labels, predicted_labels )
    
    print "Finished evaluating default Decision Tree classifier.\n"
    
    print "Default Decision Tree classifier evaluation"
    
    print "error rate (%%) = %0.6f" % error_rate
    print evaluation_report
    
# train_decision_tree_3_1_a()


# In[13]:

"""
Exercise 3.1(b)
"""
def train_decision_tree_3_1_b( criterion=None, max_depth=None, max_features=None ):
    
    if criterion == None:
        criterion_list = [ "gini", "entropy" ]
    else:
        criterion_list = [ criterion ]
    
    if max_depth == None:
        max_depth_min = 5
        max_depth_max = 25 
    else:
        max_depth_min = max_depth
        max_depth_max = max_depth
        
    max_depth_step = 20
        
    if max_features == None:
        max_features_min = 28*28-400
        max_features_max = 28*28
    else:
        max_features_min = max_features
        max_features_max = max_features
    
    max_features_step = 200
    
    least_error = 101.0
    least_error_parameters = ""
    
    for max_depth in xrange( max_depth_min, max_depth_max + 1, max_depth_step ):

        for max_features in xrange( max_features_min, max_features_max + 1, max_features_step ):
            
            for criterion in criterion_list:
    
                #print "Started learning custom Decision Tree classifier.\n"

                clf = tree.DecisionTreeClassifier( 
                                        criterion=criterion,
                                        max_depth=max_depth,
                                        max_features=max_features
                                        )
                clf.fit( array_training_images, array_training_labels )

                #print "Finished learning custom Decision Tree classifier.\n"

                #print "Started evaluating custom Decision Tree classifier.\n"

                predicted_labels = clf.predict( array_test_images )

                error_rate = 100.0 * (1.0 - accuracy_score( array_test_labels, predicted_labels ))
                evaluation_report = classification_report( array_test_labels, predicted_labels )

                parameters = ("criterion = %s" % criterion + 
                              "\nmax_depth = %i" % max_depth + 
                              "\nmax_features = %i" % max_features)
                
                if( error_rate < least_error ):
                    least_error_parameters = parameters
                    least_error = error_rate
                    least_error_criterion = criterion
                    least_error_max_depth = max_depth
                    least_error_max_features = max_features
                
                #print "Finished evaluating custom Decision Tree classifier.\n"

                print "\n\nCustom Decision Tree classifier evaluation\n"
                print parameters
                
                print "\nerror rate (%%) = %0.6f" % error_rate

                print "\n--------------\n\nBest parameters:\n"
                print "error rate (%%) = %0.6f" % least_error
                print least_error_parameters
                
                # print evaluation_report

    clf = tree.DecisionTreeClassifier( 
                                    criterion=least_error_criterion,
                                    max_depth=least_error_max_depth,
                                    max_features=least_error_max_features
                                    )
    clf.fit( array_training_images, array_training_labels )
    return clf

# Try different parameter combinations
# train_decision_tree_3_1_b()


# In[14]:

# Learn a decision tree given specific parameters
clf_tree_custom = train_decision_tree_3_1_b( criterion="entropy", max_depth = 30, max_features = 784 )

#dot_data = StringIO() 
#tree.export_graphviz(clf, out_file=dot_data) 
#graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
#graph.write_pdf(filepath + "decision_tree_3_1_b.pdf") 

"""
Exercise 3.1(d) Visualize pixel importances
"""
importances = clf_tree_custom.feature_importances_ / max(clf_tree_custom.feature_importances_)
plot_image( importances * 255 )


# In[15]:

from sklearn.ensemble import RandomForestClassifier

"""
Exercise 3.2(a)
"""
def train_random_forest_3_2_a():
    
    print "Started learning default Random Forest classifier.\n"
    
    clf = RandomForestClassifier()
    clf.fit( array_training_images, array_training_labels )
    
    print "Finished learning default Random Forest classifier.\n"
    
    print "Started evaluating default Random Forest classifier.\n"
    
    predicted_labels = clf.predict( array_test_images )
    
    error_rate = 100.0 * (1.0 - accuracy_score( array_test_labels, predicted_labels ))
    evaluation_report = classification_report( array_test_labels, predicted_labels )
    
    print "Finished evaluating default Random Forest classifier.\n"
    
    print "Default Random Forest classifier evaluation"
    
    print "error rate (%%) = %0.6f" % error_rate
    print evaluation_report
    
# train_random_forest_3_2_a()


# In[16]:

"""
Exercise 3.2(b)
"""
def train_random_forest_3_2_b( n_estimators=None, criterion=None, max_depth=None, max_features=None ):
    
    if n_estimators == None:
        n_estimators_range = xrange( 10, 20+1, 5 )
    else:
        n_estimators_range = [n_estimators]
    
    if criterion == None:
        criterion_list = [ "gini", "entropy" ]
    else:
        criterion_list = [ criterion ]
    
    if max_depth == None:
        max_depth_min = 10
        max_depth_max = 30 
    else:
        max_depth_min = max_depth
        max_depth_max = max_depth
        
    max_depth_step = 10
        
    if max_features == None:
        max_features_min = 28*28-400
        max_features_max = 28*28
    else:
        max_features_min = max_features
        max_features_max = max_features
    
    max_features_step = 200
    
    least_error = 101.0
    least_error_parameters = ""
    
    for n_estimators in n_estimators_range:
    
        for max_depth in xrange( max_depth_min, max_depth_max + 1, max_depth_step ):

            for max_features in xrange( max_features_min, max_features_max + 1, max_features_step ):

                for criterion in criterion_list:

                    #print "Started learning custom Decision Tree classifier.\n"

                    clf = RandomForestClassifier( 
                                            n_estimators=n_estimators,
                                            criterion=criterion,
                                            max_depth=max_depth,
                                            max_features=max_features
                                            )
                    clf.fit( array_training_images, array_training_labels )

                    #print "Finished learning custom Decision Tree classifier.\n"

                    #print "Started evaluating custom Decision Tree classifier.\n"

                    predicted_labels = clf.predict( array_test_images )

                    error_rate = 100.0 * (1.0 - accuracy_score( array_test_labels, predicted_labels ))
                    evaluation_report = classification_report( array_test_labels, predicted_labels )

                    parameters = ("n_estimators = %i" % n_estimators +  
                                  "\ncriterion = %s" % criterion + 
                                  "\nmax_depth = %i" % max_depth + 
                                  "\nmax_features = %i" % max_features)

                    if( error_rate < least_error ):
                        least_error_parameters = parameters
                        least_error = error_rate
                        least_error_n_estimators = n_estimators
                        least_error_criterion = criterion
                        least_error_max_depth = max_depth
                        least_error_max_features = max_features

                    #print "Finished evaluating custom Decision Tree classifier.\n"

                    print "\n\nCustom Random Forest classifier evaluation\n"
                    print parameters

                    print "\nerror rate (%%) = %0.6f" % error_rate

                    print "\n--------------\n\nBest parameters:\n"
                    
                    print least_error_parameters
                    
                    print "\nerror rate (%%) = %0.6f\n" % least_error

                    # print evaluation_report

    clf = RandomForestClassifier( 
                                    n_estimators=n_estimators,
                                    criterion=least_error_criterion,
                                    max_depth=least_error_max_depth,
                                    max_features=least_error_max_features
                                    )
    clf.fit( array_training_images, array_training_labels )
    return clf

# Try different parameter combinations
# clf_forest = train_random_forest_3_2_b()


# In[17]:

clf_forest_custom = train_random_forest_3_2_b( n_estimators=20, criterion="entropy", max_depth=30, max_features=584 )

"""
Exercise 3.2(c) Visualize pixel importances
"""
importances = clf_forest_custom.feature_importances_ / max(clf_forest_custom.feature_importances_)
plot_image( importances*255 )


# In[ ]:

"""
Exercise 3.3 Cross-Validation
"""

from sklearn import cross_validation

def print_cv_score(scores):
    print("Error rate (%%): %i (+/- %i)" % ( (1.0-scores.mean()) * 100, scores.std() * 2 * 100))

cross_validations = 5

tree_custom_scores = cross_validation.cross_val_score( clf_tree_custom, array_training_images, array_training_labels, cv = cross_validations )
print_cv_score(tree_custom_scores)

forest_custom_scores = cross_validation.cross_val_score( clf_forest_custom, array_training_images, array_training_labels, cv = cross_validations )
print_cv_score(forest_custom_scores)


# In[ ]:




