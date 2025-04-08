from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

main = Tk()
main.title("Location prediction on Twitter using machine learning Techniques")
main.geometry("1300x1200")

global filename
global X, Y
global X_train, X_test, y_train, y_test
global tfidf_vectorizer
accuracy = []

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

textdata = []
labels = []
global classifier

location_name = ['Arizona', 'Brazil', 'Brooklyn', 'Chennai', 'Florida', 'India', 'Indonesia',
                 'Kerala', 'Kirkwall', 'Pune', 'Sweden', 'United States', 'mexico', 'uk']

def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def uploadDataset():    
    global filename
    text.delete('1.0', END)
    le = LabelEncoder()
    filename = filedialog.askopenfilename(initialdir="Dataset")
    textdata.clear()
    labels.clear()
    dataset = pd.read_csv(filename)
    print(np.unique(dataset['location']))
    dataset['location'] = pd.Series(le.fit_transform(dataset['location'].astype(str)))
    print(np.unique(dataset['location']))
    for i in range(len(dataset)):
        msg = dataset.get_value(i, 'text')
        label = dataset.get_value(i, 'location')
        msg = str(msg)
        msg = msg.strip().lower()
        labels.append(label)
        clean = cleanPost(msg)
        textdata.append(clean)
        text.insert(END,clean+"\n")
    


def preprocess():
    text.delete('1.0', END)
    global X, Y
    global tfidf_vectorizer
    global X_train, X_test, y_train, y_test
    stopwords=stopwords = nltk.corpus.stopwords.words("english")
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords, use_idf=True, ngram_range=(1,2),smooth_idf=False, norm=None, decode_error='replace')
    tfidf = tfidf_vectorizer.fit_transform(textdata).toarray()        
    df = pd.DataFrame(tfidf, columns=tfidf_vectorizer.get_feature_names())
    text.insert(END,str(df))
    print(df.shape)
    df = df.values
    X = df[:, 0:df.shape[1]]
    Y = np.asarray(labels)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    print(X)
    print(Y)
    print(Y.shape)
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"\n\nTotal tweets found in dataset : "+str(len(X))+"\n")
    text.insert(END,"Total tweets used to train machine learning algorithms : "+str(len(X_train))+"\n")
    text.insert(END,"Total tweets used to test machine learning algorithms  : "+str(len(X_test))+"\n")

def runML():
    global X, Y
    global tfidf_vectorizer
    global classifier
    global X_train, X_test, y_train, y_test
    global accuracy
    accuracy.clear()
    text.delete('1.0', END)

    cls = GaussianNB()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test) 
    a = accuracy_score(y_test,predict)*100
    accuracy.append(a)
    text.insert(END,"Naive Bayes Accuracy : "+str(a)+"\n\n")

    cls = SVC()
    cls.fit(X, Y)
    predict = cls.predict(X_test) 
    a = accuracy_score(y_test,predict)*100
    accuracy.append(a)
    text.insert(END,"SVM Accuracy : "+str(a)+"\n\n")

    cls = DecisionTreeClassifier()
    cls.fit(X, Y)
    predict = cls.predict(X_test) 
    a = accuracy_score(y_test,predict)*100
    accuracy.append(a)
    text.insert(END,"Decision Tree Accuracy : "+str(a)+"\n\n")
    classifier = cls

    
def graph():
    height = accuracy
    bars = ('Naive Bayes','SVM','Decision Tree')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title('Accuracy Comparison Graph')
    plt.show()

def predict():
    global tfidf_vectorizer
    global classifier
    testfile = filedialog.askopenfilename(initialdir="Dataset")
    testData = pd.read_csv(testfile)
    text.delete('1.0', END)
    testData = testData.values
    print(testData)
    for i in range(len(testData)):
        msg = testData[i]
        msg1 = testData[i]
        msg = msg[0]
        print(msg)
        review = msg.lower()
        review = review.strip().lower()
        review = cleanPost(review)
        testReview = tfidf_vectorizer.transform([review]).toarray()
        predict = classifier.predict(testReview)[0]
        print(predict)
        text.insert(END,str(msg1)+" === LOCATION PREDICTED AS "+location_name[predict]+"\n\n")
        
    
font = ('times', 15, 'bold')
title = Label(main, text='Location prediction on Twitter using machine learning Techniques')
title.config(bg='gold2', fg='thistle1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Tweets Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset", command=preprocess)
processButton.place(x=20,y=150)
processButton.config(font=ff)

dtButton = Button(main, text="Run Machine Learning Algorithm", command=runML)
dtButton.place(x=20,y=200)
dtButton.config(font=ff)

graphButton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphButton.place(x=20,y=250)
graphButton.config(font=ff)

predictButton = Button(main, text="Predict Location from Test Tweets", command=predict)
predictButton.place(x=20,y=300)
predictButton.config(font=ff)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=330,y=100)
text.config(font=font1)

main.config(bg='DarkSlateGray1')
main.mainloop()
