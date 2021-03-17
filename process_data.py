import os
import json
import nltk, re, pprint
from bs4 import BeautifulSoup
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

nltk.download('punkt')
from nltk import word_tokenize

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

texts = []
statuses = []


def process(subfolder):
    directory = 'data/' + subfolder
    for file in os.listdir(directory):
        with open(directory + '/' + file) as json_file:
            page = json.load(json_file)
            text = process_text(page['content'])
            texts.append(text)
            statuses.append(page['status'])


def process_text(text):
    # Removendo Tags e conteúdos do HTML
    soup = BeautifulSoup(text)
    text = soup.get_text()

    # Removendo todos os caracteres desnecessários
    text = re.sub('[^A-Za-z]', ' ', text)

    # Padronizando Case
    text = text.lower()

    # Tokenizando texto
    tokens = word_tokenize(text)

    # Removendo Stopwords
    for word in tokens:
        if word in stopwords.words('english'):
            tokens.remove(word)

    # Realizando Stemming
    for i in range(len(tokens)):
        tokens[i] = stemmer.stem(tokens[i])

    return ' '.join(tokens)


def bag_of_words():
    matrix = CountVectorizer(max_features=1000)
    X = matrix.fit_transform(texts).toarray()
    return train_test_split(X, statuses)


def naive_bayes(X_train, X_test, y_train, y_test):
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    print_metrics(y_pred, y_test, 'Naive Bayes')


def support_vector_machine(X_train, X_test, y_train, y_test):
    classifier = svm.SVC(kernel='linear')  # Linear Kernel
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    print_metrics(y_pred, y_test, 'Support Vector Machine')


def decision_trees(X_train, X_test, y_train, y_test):
    classifier = DecisionTreeClassifier()
    classifier = classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    print_metrics(y_pred, y_test, 'Decision Trees (JV8)')


def logistic_regression(X_train, X_test, y_train, y_test):
    classifier = LogisticRegression()
    classifier = classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    print_metrics(y_pred, y_test, 'Logistic Regression')


def multilayer_perceptron(X_train, X_test, y_train, y_test):
    classifier = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    classifier = classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    print_metrics(y_pred, y_test, 'MultiLayer Perceptron')


def print_metrics(y_pred, y_test, classification_method):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("################################## " + classification_method + " ##################################")
    print('Accuracy: ' + str(accuracy))
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))


if __name__ == '__main__':
    process('sambafoot')
    X_train, X_test, y_train, y_test = bag_of_words()
    naive_bayes(X_train, X_test, y_train, y_test)
    time.sleep(120)
    decision_trees(X_train, X_test, y_train, y_test)
    time.sleep(120)
    support_vector_machine(X_train, X_test, y_train, y_test)
    time.sleep(120)
    logistic_regression(X_train, X_test, y_train, y_test)
    time.sleep(120)
    multilayer_perceptron(X_train, X_test, y_train, y_test)
