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
  #  result = classifier.classify_nb(docs_test_2, labels_test_2)
 #   print("data_test_2 classified as ", result[0])
  #  print("data_test_2 is actually", labels_test_2)

    #Task 3
    print("\n Testing the given test data:")
    prediction = classifier.classify_documents(training_docs)
    accuracy = sum(1 for i in range(len(prediction)) if prediction[i] == training_labels[i]) / float(len(prediction))
    print("Accuracy on the given test data is {0:.4f}".format(accuracy))

    write_data_to_file("data_word_model.txt", classifier.data_word_model)
    write_data_to_file("data_model.txt", classifier.data_model)
    write_results_to_file("data_results.txt", classifier.data_results)

    accuracy_results = open("accuracy_results.txt", "w")
    accuracy_results.write("Accuracy on the given test data is {0:.4f}".format(accuracy))
    accuracy_results.close()

main()