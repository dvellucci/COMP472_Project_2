import FileReader
from collections import Counter
import ClassifierModule

def write_data_to_file(file, dictionary):
    data_file = open(file, "w")
    for key in dictionary:
        value = dictionary[key]
        data_file.write(key + '{}\n'.format(value))
    data_file.close()

def write_results_to_file(file, dictionary):
    data_file = open(file, "w")
    for key in dictionary:
        value = dictionary[key]
        data_file.write(key + '{}\n'.format(value) + '\n')
    data_file.close()

def main(): 

    docs, labels = FileReader.read_documents("data")

    split_point = int(0.80*len(docs))

    training_docs = docs[:split_point]
    training_labels = labels[:split_point]
 
    evaluation_docs = docs[split_point:]
    evaluation_labels = labels[split_point:]

    classifier = ClassifierModule.Classifier()

    #task 1
    #Calculate log class priors and retrieve count of each word appearing in each label
    word_counts = classifier.train_nb(training_docs, training_labels)

    #task 2
    #probability of each word appearing in each class
    result = classifier.score_doc_label(training_docs, training_labels)

    #try to classify a few short documents
    docs_test_1, labels_test_1 = FileReader.read_documents("data_test_1")
  #  result = classifier.classify_nb(docs_test_1, labels_test_1)
  #  print("data_test_1 classified as ", result[0])
  #  print("data_test_1 is actually", labels_test_1)

    docs_test_2, labels_test_2 = FileReader.read_documents("data_test_2")
    #result = classifier.classify_nb(docs_test_2, labels_test_2)
    #print("data_test_2 classified as ", result[0])
    #print("data_test_2 is actually", labels_test_2)

    docs_test_3, labels_test_3 = FileReader.read_documents("data_test_3")
    #result = classifier.classify_nb(docs_test_3, labels_test_3)
    #print("data_test_3 classified as ", result[0])
    #print("data_test_3 is actually", labels_test_3)

    #Task 3 - Evaluate the test data
    print("\n Testing the given test data:")
    prediction = classifier.classify_documents(evaluation_docs, evaluation_labels)
    accuracy = sum(1 for i in range(len(prediction)) if prediction[i] == evaluation_labels[i]) / float(len(prediction))
    print("Accuracy on the given test data is {0:.4f}".format(accuracy))

    write_data_to_file("data_word_model_probabilities.txt", classifier.data_word_model_probabilities)
    write_data_to_file("data_word_model_scores.txt", classifier.data_word_model_scores)
    write_results_to_file("data_document_results.txt", classifier.data_document_results)

    accuracy_results = open("accuracy_results.txt", "w")
    accuracy_results.write("Accuracy on the given test data is {0:.4f}".format(accuracy))
    accuracy_results.close()

main()


#FOR task 4: Finding mis-classified documents
#To possibly see why these were mis-classified, it could be helpful to look at the data_word_model_scores.txt or data_word_model_probabilities files to see the scores/probabilities of each word
'''
The document i followed all instructions before i attempted to do an upgrade from windows me to windows xp - infact , the more i read the more i was told to do a clean upgrade . 
so , i did a backup and removed programs that were not compatible and i got some instructions off the internet on how to do it and it went very smooth . 
i love what it gave me . clean upgrade means a clean start - you remove everything and start with a new program" was incorrectly classified as NEG.

The document "this is a difficult book for beginner-level spanish language students .
i kept it b / c i understand that it is one they use at the immersion course i will be taking soon in mexico . probably will not use it before that or subsequent"
was incorrectly classified as POS

The document "i really enjoyed this movie ! its not one of the best and a little cheesy ! 
but easy to watch and one of the funniest films ive seen him in ! just a shame he never got to show us what a good actor he could have been"
was mis-classified as NEG.

The document "i really enjoyed this movie ! its not one of the best and a little cheesy ! but easy to watch and one of the funniest 
films ive seen him in ! just a shame he never got to show us what a good actor he could have been" was mis-classified as NEG
'''