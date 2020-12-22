# Span mail detection
# Ham mails 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


dataset = pd.read_csv('spamham.csv')
# print(dataset)
# replace null values with null string

mail_data = dataset.where((pd.notnull(dataset)), '')

print(mail_data.shape)
print(mail_data.head())

# label sapm mail as 0 and ham is 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

print(mail_data.head())

# separate the data text and labels
X = mail_data['Message']
y = mail_data['Category']
print(X)
print(y)

# split into trianing data and testing data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2,random_state = 1)

# featuer extraction ---> with TfidVectorizer
# and convert all text to lower case
feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase = 'True')
X_train = feature_extraction.fit_transform(X_train)
X_test = feature_extraction.transform(X_test)

print(X_train)
print(X_test)

# convert y_train and y_test into integers
y_train = y_train.astype('int')
y_test = y_test.astype('int')

# training the model
classifier = LinearSVC()
classifier.fit(X_train, y_train)
predict_on_trainig_data = classifier.predict(X_train)
accuracy_on_trainig_data = accuracy_score(y_train, predict_on_trainig_data)
print(accuracy_on_trainig_data)

print(X_test.shape, y_test.shape)
print(X_train.shape, y_train.shape)

predict_on_test_data = classifier.predict(X_test)
accuracy_on_test_data = accuracy_score(y_test, predict_on_test_data)
print(accuracy_on_test_data)

#predict new mail and predict

input_mail = ["WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."]

# convert text to feature
# input_mail_features = input_mail.reshape[1,-1]

input_mail_features = feature_extraction.transform(input_mail)
pre = classifier.predict(input_mail_features)
print(pre)

if pre[0] == 1:
	print("Ham mail")
else:
	print("Spam")