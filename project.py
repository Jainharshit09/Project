from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import collections
pip install sklearn-som
from sklearn_som.som import SOM
pip install scikit-learn-extra
pip install sklearn-som
pip install fuzzy-c-means
from sklearn_extra.cluster import KMedoids
from sklearn_som.som import SOM
from fcmeans import FCM
pip install opendatasets --upgrade --quiet
import opendatasets as od
od.download('https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones')
train_file = './human-activity-recognition-with-smartphones/train.csv'
test_file = './human-activity-recognition-with-smartphones/test.csv'
train_df = pd.read_csv(train_file)
train_df
test_df = pd.read_csv(test_file)
test_df
print(train_df.info())
print()
print(test_df.info())

def set_target(df) :
  l = []
  for x in df['Activity'] :
    if x == 'WALKING' :
      l.append(1)
    elif x == 'WALKING_UPSTAIRS' :
      l.append(2)
    elif x == 'WALKING_DOWNSTAIRS' :
      l.append(3)
    elif x == 'SITTING' :
      l.append(4)
    elif x == 'STANDING' :
      l.append(5)
    else :
      l.append(6)
  return l

train_df['target'] = set_target(train_df)
test_df['target'] = set_target(test_df)

df = pd.concat([train_df,test_df])

train_df.corr()['target']

features = ['tBodyAcc-mean()-X','tBodyAcc-mean()-Y','tBodyAcc-mean()-Y','tBodyAcc-mean()-Z','tBodyAcc-std()-X','tBodyAcc-std()-Y',
 'tBodyAcc-std()-Z','fBodyAcc-mean()-X','fBodyAcc-mean()-Y','fBodyAcc-mean()-Z','fBodyAcc-std()-X', 'fBodyAcc-std()-Y',
 'fBodyAcc-std()-Z','angle(tBodyAccMean,gravity)','angle(tBodyAccJerkMean),gravityMean)','angle(tBodyGyroMean,gravityMean)',
 'angle(tBodyGyroJerkMean,gravityMean)','angle(X,gravityMean)','angle(Y,gravityMean)','angle(Z,gravityMean)']

X = train_df[features]
y = train_df['target']
X_test = test_df[features]
y_test = test_df['target']

[x for x in train_df if train_df.isna == True]
[x for x in test_df if test_df.isna == True]

y_train=train_df.iloc[:,-2]
Category_count=np.array(y_train.value_counts())
activity=sorted(y_train.unique())
activity

plt.figure(figsize=(16,6))
plt.pie(Category_count,labels=activity, autopct = '%0.2f')

fig = plt.figure(figsize = (10, 6))
ax = fig.add_axes([0,0,1,1])
ax.set_title("Count of each activity", fontsize = 15)
plt.tick_params(labelsize = 10)
sns.countplot(x = "Activity", data = train_df)
for i in ax.patches:
    ax.text(x = i.get_x() + 0.2, y = i.get_height()+10, s = str(i.get_height()), fontsize = 20, color = "grey")
plt.xlabel('')
plt.ylabel("Count", fontsize = 15)
plt.tick_params(labelsize = 13)
plt.xticks(rotation = 40)
plt.show()

fig = plt.figure(figsize = (12, 8))
ax = fig.add_axes([0,0,1,1])
ax.set_title("Activity by each test subject", fontsize = 15)
plt.tick_params(labelsize = 15)
sns.countplot(x = "subject", hue = "Activity", data = train_df)
plt.xlabel("Subject ID", fontsize = 15)
plt.ylabel("Count", fontsize = 15)
plt.show()

clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

predicted = clf.predict(X)
report = classification_report(train_df['target'],predicted,digits=4, output_dict=True)
fi = pd.DataFrame(report).transpose()
fi.to_csv('result.csv', mode='a', header=False)
fi
predicted = clf.predict(X_test)
print(classification_report(y_test,predicted,digits=4))

import numpy as np
x = np.array(df[features])
y = np.array(df['target'])
x, y = shuffle(x, y, random_state=100)

def kfold(features,y_actual):
  kf = KFold(n_splits=5,random_state=1000, shuffle=True)
  kf.get_n_splits(features)
  all_x_train = []
  all_x_test = []
  all_y_train = []
  all_y_test = []
  for train_index, test_index in kf.split(features):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = y_actual[train_index], y_actual[test_index]
    all_x_train.append(X_train)
    all_x_test.append(X_test)
    all_y_train.append(y_train)
    all_y_test.append(y_test)
  all_x_train, all_x_test, all_y_train, all_y_test  = np.array(all_x_train), np.array(all_x_test), np.array(all_y_train), np.array(all_y_test)
  for i in range(0, 5):
    all_y_train[i] = all_y_train[i].flatten()
  return all_x_train, all_x_test, all_y_train, all_y_test

import warnings
warnings.filterwarnings('ignore')

all_x_train, all_x_test, all_y_train, all_y_test = kfold(x, y)

for i in range(0, 5):

  print('\n')
  print("For fold no:", i+1)
  print()

  logistic_reg_classifier = LogisticRegression(penalty = 'none', random_state=67, solver = 'sag', max_iter = 100).fit(all_x_train[i], all_y_train[i])

  #Accuracy for train set
  print("Accuracy on training data: " + str(logistic_reg_classifier.score(all_x_train[i], all_y_train[i])))
  predicted = logistic_reg_classifier.predict(all_x_test[i])

  print("Testing Accuracy Score: " + str(accuracy_score(all_y_test[i], predicted)*100))

  #confusion matrix using heat map
  print('Confusion Matrix : \n')
  array=confusion_matrix(all_y_test[i], predicted)
  df_cm = pd.DataFrame(array, index = [i for i in "012345"],columns = [i for i in "012345"])
  plt.figure(figsize = (10,7))
  sns.heatmap(df_cm, annot=True)
  plt.show()

  print("\nClassification Report: ")
  report = classification_report(all_y_test[i], predicted, labels =[1,2,3,4,5,6], digits=5, output_dict=True)
  fi = pd.DataFrame(report).transpose()
  fi.to_csv('result.csv', mode='a', header=False)
  display(fi)
  print()

from sklearn.linear_model import Perceptron
for i in range(0, 5): # for 5 fold

  print("\nFor fold no:", i+1)
  print()

  slp = Perceptron(random_state=1,penalty='elasticnet')
  slp.fit(all_x_train[0], all_y_train[0])

  print("Accuracy on training data: " + str(slp.score(all_x_train[i], all_y_train[i])))
  predicted = slp.predict(all_x_test[i])

  print("Testing Accuracy Score: " + str(accuracy_score(all_y_test[i], predicted)*100))

  #confusion matrix using heat map
  print('Confusion Matrix : \n')
  array=confusion_matrix(all_y_test[i], predicted)
  df_cm = pd.DataFrame(array, index = [i for i in "012345"],columns = [i for i in "012345"])
  plt.figure(figsize = (10,7))
  sns.heatmap(df_cm, annot=True)
  plt.show()

  print("Classification Report : ")
  report = classification_report(all_y_test[i], predicted, labels=[1, 2, 3, 4, 5, 6], digits=5,output_dict=True)
  fi = pd.DataFrame(report).transpose()
  fi.to_csv('result.csv', mode='a', header=False)
  display(fi)
  print()
for i in range(0, 5):

  print("\nFor fold no:", i+1)
  print()

  mlp = MLPClassifier(hidden_layer_sizes=(100), activation='relu',solver='lbfgs', random_state=1, max_iter=1000).fit(all_x_train[0], all_y_train[0])
  mlp.predict(all_x_train[0])

  print("Accuracy on training data: " + str(mlp.score(all_x_train[i], all_y_train[i])))
  predicted = mlp.predict(all_x_test[i])

  print("Testing Accuracy Score: " + str(accuracy_score(all_y_test[i], predicted)))

  print('Confusion Matrix : \n')
  array=confusion_matrix(all_y_test[i], predicted)
  df_cm = pd.DataFrame(array, index = [i for i in "012345"],columns = [i for i in "012345"])
  plt.figure(figsize = (10,7))
  sns.heatmap(df_cm, annot=True)
  plt.show()

  print("Classification Report : ")
  labels = [1,2,3,4,5,6]
  report = classification_report(all_y_test[i], predicted, labels=labels, digits=5, output_dict=True)
  fi = pd.DataFrame(report).transpose()
  fi.to_csv('result.csv', mode='a', header=False)
  display(fi)
  print()



kmean = KMeans(n_clusters=6)
kmean.fit(X)
kmean.cluster_centers_
y_kmeans = kmean.predict(X)
silhouette_score(X, y_kmeans)
kmean.inertia_
y_kmeans = kmean.predict(X_test)
silhouette_score(X_test, y_kmeans)
y_kmeans = kmean.predict(x)
silhouette_score(x, y_kmeans)
kme = [dict(collections.Counter(y_kmeans))[x] for x in range(6)]

plt.figure(figsize=(16,6))
plt.pie(kme,labels= set(y_kmeans), autopct = '%0.2f')

error = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(X)
    error.append(kmeans.inertia_)
plt.plot(range(1, 10), error)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('error')
plt.show()

kmedoids = KMedoids(n_clusters=6, random_state=0).fit(X)
y_kmed = kmedoids.predict(X)
silhouette_score(X, y_kmed)
kmedoids.inertia_
y_kmed = kmedoids.predict(X_test)
silhouette_score(X_test, y_kmed)
y_kmed = kmedoids.predict(x)
silhouette_score(x, y_kmed)
kmed = [dict(collections.Counter(y_kmed))[x] for x in range(6)]

plt.figure(figsize=(16,6))
plt.pie(kmed,labels= set(y_kmed), autopct = '%0.2f')
error = []
for i in range(1, 10):
    kmedoid = KMedoids(n_clusters = i, random_state = 42)
    kmedoid.fit(X)
    error.append(kmedoid.inertia_)
plt.plot(range(1, 10), error)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('error')
plt.show()fcm = FCM(n_clusters=6)
fcm.fit(x)
y_fcm=fcm.predict(x)
y_fcm
silhouette_score(x, y_fcm)
def calc_sse(centers,x,y):
  sse=0
  for i in range(x.shape[0]):
    sse+=np.sum((x[i]-centers[y[i]])**2)

  return sse
  calc_sse(fcm.centers,x,y_fcm)s
