import numpy as np
import matplotlib.pyplot as plt
import caffe
from scipy.stats import entropy
from operator import itemgetter

caffe_root = '../'

MODEL_FILE = '../models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = '../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

IMAGE_FILENAMES = [ 'images/IMG_%i.jpg' % i for i in xrange(1,9) ]

import re
def load_class_names():
	with open("../data/ilsvrc12/synset_words.txt") as f:
		class_names = f.readlines()
	for i in xrange(len(class_names)):
		class_names[i] = re.sub( "(^[A-Za-z0-9]*\s+)|(\n$)","",class_names[i] )

	return class_names

class_names = load_class_names()

caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

def sort_tuple_list_by_item( tuple_list, idx_item, descend=False ):

    return sorted( tuple_list, key=itemgetter(idx_item), reverse=descend )

def load_image_and_predict_class(image_filename):

	input_image = caffe.io.load_image(image_filename)
	prediction = net.predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically

	print '%s\n' % image_filename

	if prediction[0].shape[0] != 1000:

		print "Error: Prediction does not contain all 1000 classes."
		return

	top3 = sort_tuple_list_by_item( zip(class_names, prediction[0]), 1, descend=True )[:3]

	for i in xrange(3):
		print 'predicted class %i: %s' % ( i+1, top3[i][0] )
		print 'probability: %f' % top3[i][1]

	print '\nentropy: %f' % entropy(prediction[0])

	print '\n'

for image_filename in IMAGE_FILENAMES:
	load_image_and_predict_class( image_filename )