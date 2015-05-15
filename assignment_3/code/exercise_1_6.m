%% please run load_mnist before executing this script

%% Exercises 1.6: Train on whole dataset

samples = 60000;

error_rate = train_and_evaluate( samples, images_training, labels_training, images_test, labels_test, 'AdaBoostM2', 'Tree' );