import FileReader
from collections import Counter
import ClassifierModule

def main(): 

    docs, labels = FileReader.read_documents("data")

    split_point = int(0.80*len(docs))

    training_docs = docs[:split_point]
    training_labels = labels[:split_point]
 
    evaluation_docs = docs[split_point:]
    evaluation_labels = labels[split_point:]

    classifier = ClassifierModule.Classifier()

    #Task 3
    word_counts = classifier.train_nb(training_docs, training_labels)
    prediction = classifier.classify_documents(training_docs)
    accuracy = sum(1 for i in range(len(prediction)) if prediction[i] == training_labels[i]) / float(len(prediction))
    print("Accuracy for classifier is {0:.4f}".format(accuracy))

main()