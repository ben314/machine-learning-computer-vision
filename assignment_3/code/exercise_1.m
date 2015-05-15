%% please run load_mnist before executing this script

%% Exercises 1.3, 1.4, 1.5: Default AdaBoost and Grid search

samples = 1000;

error_rates = [];
error_rates = [error_rates, train_and_evaluate_collective( samples, images_training, labels_training, images_test, labels_test, 'AdaBoostM2', 'Tree' )];
%error_rates = [error_rates, train_and_evaluate_collective( samples, images_training, labels_training, images_test, labels_test, 'AdaBoostM2', 'Discriminant' )];
error_rates = [error_rates, train_and_evaluate_collective( samples, images_training, labels_training, images_test, labels_test, 'LPBoost', 'Tree' )];
%error_rates = [error_rates, train_and_evaluate_collective( samples, images_training, labels_training, images_test, labels_test, 'LPBoost', 'Discriminant' )];

plot( error_rates );
xlabel( 'learning rounds' );
ylabel( 'error rate' );
%legend( 'AdaBoostM2 tree', 'AdaBoostM2 discriminant', 'LPBoost tree', 'LPBoost discriminant' );
legend( 'AdaBoostM2 tree', 'LPBoost tree' );