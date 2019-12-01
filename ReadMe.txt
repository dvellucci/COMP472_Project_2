The file ClassifierModule is responsible for all the functions needed to calculate the probabilities and scores of documents.

TASK 1
The function train_nb finds the prior probabilities of each class (NEG and POS) and returns the number of times each word appears in each class (the word_counts object).

TASK 2
The function score_doc_label finds the log probabilities of each word appearing in each label. 

TASK 3
The function classify_nb will classify a new document using the prior log class probabilities from train_nb and the probabilites of the words found in score_doc_label 
to compute the score for each line in a document.

The function classify_documents is used for classifying the given documents of the project. It will compute the POS and NEG score for each line and store whichever one is bigger. 


We wrote some information to text files.

The file data_word_model_probabilities contains every unique word and their log probabilities for each class.
The file data_word_model_scores contains every unique word and their scores for each class.
The file data_document_results contains every line in the document, their computed scores, what they were classified as and the actual class they belong to so you can see which ones were classified correctly. 

The file accuracy_results will have the accuracy of the classification.


To run the document, run the PolarityEstimation.py file.

I split up parts of the code with comments showing which parts are for which task.
You can use comments to only run the parts that you would like to run. 