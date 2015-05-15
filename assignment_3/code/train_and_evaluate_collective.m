function [ error_rates ] = train_and_evaluate_collective( samples, images_training, labels_training, images_test, labels_test, Method, Learners )

i_max = 10;
error_rates = zeros(i_max,1);

parfor i = 1:i_max

    learning_rounds = i * 100;
    
    ensemble_model = fitensemble(images_training(1:samples,:),labels_training(1:samples),Method,learning_rounds,Learners);
    
    labels_predicted = predict( ensemble_model, images_test );
    
    error_rates(i) = evaluate_prediction( labels_test, labels_predicted );
    
    fprintf( 'learners:         %s\n', Learners );
    fprintf( 'training samples: %i\n', samples );
    fprintf( 'learning rounds:  %i\n', learning_rounds );
    fprintf( 'error rate:       %.2f\n\n', error_rates(i) );

end

end