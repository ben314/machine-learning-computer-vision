function error_rate = train_and_evaluate( samples, images_training, labels_training, images_test, labels_test, Method, Learners )

learning_rounds = 800;

ensemble_model = fitensemble(images_training(1:samples,:),labels_training(1:samples),Method,learning_rounds,Learners);

labels_predicted = predict( ensemble_model, images_test );

error_rate = evaluate_prediction( labels_test, labels_predicted );

fprintf( 'learners:         %s\n', Learners );
fprintf( 'training samples: %i\n', samples );
fprintf( 'learning rounds:  %i\n', learning_rounds );
fprintf( 'error rate:       %.2f\n\n', error_rate );

end