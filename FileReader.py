import codecs 

def read_documents(file):
    docs = []
    labels = []

    with open(file, encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            docs.append(words[3:])
            labels.append(words[1])
    return docs, labels

def main():
    docs, labels = read_documents("data")
main()