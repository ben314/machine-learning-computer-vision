mnist_path = 'mlcv/assignment_3/mnist/';

%% Exercise 1.2: Loading MNIST
%% The data is loaded utilizing the MNIST helper functions from:
%% http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset

images_training = loadMNISTImages(strcat(mnist_path, 'train-images.idx3-ubyte'));
images_training = images_training';
labels_training = loadMNISTLabels(strcat(mnist_path, 'train-labels.idx1-ubyte'));

images_test = loadMNISTImages(strcat(mnist_path, 't10k-images.idx3-ubyte'));
images_test = images_test';
labels_test = loadMNISTLabels(strcat(mnist_path, 't10k-labels.idx1-ubyte'));