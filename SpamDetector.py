import pandas as pd
from sklearn.model_selection import train_test_split #this modules for split data
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn import svm

spam = pd.read_csv(r"spam.csv") #read data from folder 
#spam is store in dataframe

#this is the training data set
z = spam['v2']#input data
y = spam["v1"]#output data

print("length of data=",len(z))

z_train, z_test,y_train, y_test = train_test_split(z,y,test_size = 0.2) #80% train  #20% test

print("length of train data",len(z_train))
print("length of test data",len(z_test))

cv = CountVectorizer()
features = cv.fit_transform(z_train)

#create models

model = svm.SVC()
model.fit(features,y_train) #models create/train

features_test = cv.transform(z_test)
print(model.score(features_test,y_test))

while True:
    msg = input("Enter Message: ")
    if msg !="q":
        msgInput = cv.transform([msg]) #msg convert into id
        #print()
        predict = model.predict(msgInput)#using id to predict its spam or ham
        print(predict)
        
        if(predict=='spam'):
            print("------------------------MESSAGE-SENT-[CHECK-SPAM-FOLDER]---------------------------")
        else:
            print("---------------------------MESSAGE-SENT-[CHECK-INBOX]------------------------------")

    else:
        break
