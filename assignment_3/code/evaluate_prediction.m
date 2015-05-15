function error_rate = evaluate_prediction( labels_test, labels_predicted )

correct_predictions = sum(labels_test == labels_predicted);
labels_total = length(labels_test);

error_rate = 100 * (1 - (correct_predictions / labels_total));

end

