import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dropout
from sklearn.preprocessing import LabelBinarizer


# naive bayes algorithm
def naiveBayes(X_train, Y_train, X_test, Y_test):
    # Create naive bayes Classifier
    gnb = GaussianNB()

    # Train the model using the training sets
    gnb.fit(X_train, Y_train)

    # Predict the response for test dataset
    Y_pred = gnb.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))


# SVM algorithm
def svmCode(X_train, Y_train, X_test, Y_test):
    # creating the SVM model and setting its parameters
    model = svm.SVC(kernel='linear', C=100, gamma=1e-7)

    # training the model
    model.fit(X_train, Y_train)

    # predicts the labels of the test images
    predicted_labels = model.predict(X_test)

    # generate accuracy of SVM
    accuracy = accuracy_score(Y_test, predicted_labels) * 100

    print("The level of accuracy is: " + str(accuracy) + "%")

    print("\nThe confusion matrix: ")
    # generate confusion matrix on results
    print(confusion_matrix(Y_test, predicted_labels))

    print("\nThe classification report: ")
    # generate classification table on results
    print(classification_report(Y_test, predicted_labels))


# logistic regression model
def logReg(X_train, Y_train, X_test, Y_test):
    classifier = LogisticRegression()
    classifier.fit(X_train, Y_train)
    score = classifier.score(X_test, Y_test)

    print("Accuracy:", score)


# linear neural network
def neuralNet(X_train, Y_train, X_test, Y_test):
    input_dim = X_train.shape[1]

    # define model
    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # fit model to the training data
    model.fit(X_train, Y_train, epochs=100, verbose=False, validation_data=(X_test, Y_test), batch_size=10)

    # evaluate accuracies for train and test set
    loss, accuracy = model.evaluate(X_train, Y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))


# convolutional neural network
def cnn(blogs_train, blogs_test, Y_train, Y_test):

    # tokenizer fit on training data
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(blogs_train)

    # blogs are turned to sequenuences by means of the tokenizer
    X_train = tokenizer.texts_to_sequences(blogs_train)
    X_test = tokenizer.texts_to_sequences(blogs_test)

    vocab_size = len(tokenizer.word_index) + 1

    maxlen = 127945

    # the data is padded so as to be of the same length throughout
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    embedding_dim = 100

    # model is defined
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))  # embeddings for words are created
    model.add(Dropout(0.2))  # dropout used for regularization
    model.add(layers.Conv1D(128, 10, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(Dropout(0.2))
    model.add(layers.Dense(10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])  # compile model
    model.summary()

    # model fit to the training data
    model.fit(X_train, Y_train,
              epochs=2,
              verbose=False,
              validation_data=(X_test, Y_test),
              batch_size=10)

    # accuracy for traininga nd test set is evaluated
    loss, accuracy = model.evaluate(X_train, Y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))


def vectorize(texts):
    # vectorizer so as to turn the words to number values
    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts)

    return vectorizer


def main():
    # load the pre processed data
    npzFile = np.load('preProcessed.npz')
    blogsM = npzFile['blogsM']
    labelsM = npzFile['labelsM']
    blogsF = npzFile['blogsF']
    labelsF = npzFile['labelsF']

    # split training and test set
    blogs_trainM, blogs_testM, labels_trainM, labels_testM = train_test_split(blogsM, labelsM, test_size=0.2,
                                                                              random_state=1000)
    blogs_trainF, blogs_testF, labels_trainF, labels_testF = train_test_split(blogsF, labelsF, test_size=0.2,
                                                                              random_state=1000)

    # join to singular training and test set
    blogs_train = np.concatenate((blogs_trainM, blogs_trainF))
    blogs_test = np.concatenate((blogs_testM, blogs_testF))
    labels_train = np.concatenate((labels_trainM, labels_trainF))
    labels_test = np.concatenate((labels_testM, labels_testF))

    vectorizer = vectorize(np.concatenate((blogsM, blogsF)))

    # change data set from words to numeric values
    X_train = vectorizer.transform(blogs_train).toarray()
    X_test = vectorizer.transform(blogs_test).toarray()
    encoder = LabelBinarizer()
    encoder.fit(labels_train)
    Y_train = encoder.transform(labels_train).ravel()
    Y_test = encoder.transform(labels_test).ravel()

    # main menu is displayed
    print("Do you wish to run: "
          "\n1) Naive Bayes Classifier"
          "\n2) Support Vector Machine"
          "\n3) Logistic Regression Model"
          "\n4) Linear Neural Network"
          "\n5) Convolutional Neural Network"
          "\nInput: ")
    choice = input()  # user input

    # accuording to the user choice, the appropriate function is carried out
    if choice == '1':
        naiveBayes(X_train, Y_train, X_test, Y_test)
    elif choice == '2':
        svmCode(X_train, Y_train, X_test, Y_test)
    elif choice == '3':
        logReg(X_train, labels_train, X_test, labels_test)
    elif choice == '4':
        neuralNet(X_train, Y_train, X_test, Y_test)
    elif choice == '5':
        cnn(blogs_train, blogs_test, Y_train, Y_test)
    else:
        print("Invalid input")


main()
