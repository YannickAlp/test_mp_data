import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

def score_classifier(dataset,classifier,labels):

    """
    performs 3 random trainings/tests to build a confusion matrix and prints results with precision and recall scores
    :param dataset: the dataset to work on
    :param classifier: the classifier to use
    :param labels: the labels used for training and validation
    :return:
    """

    kf = KFold(n_splits=3,random_state=50,shuffle=True)
    confusion_mat = np.zeros((2,2))
    recall = 0
    for training_ids,test_ids in kf.split(dataset):
        training_set = dataset[training_ids]
        training_labels = labels[training_ids]
        test_set = dataset[test_ids]
        test_labels = labels[test_ids]
        classifier.fit(training_set,training_labels)
        predicted_labels = classifier.predict(test_set)
        confusion_mat+=confusion_matrix(test_labels,predicted_labels)
        recall += recall_score(test_labels, predicted_labels)
    recall/=3
    print(confusion_mat)
    print(recall)


# Load dataset
df = pd.read_csv(".\\nba_logreg.csv")

# extract names, labels, features names and values
names = df['Name'].values.tolist() # players names
labels_no_proportion = df['TARGET_5Yrs'].values # labels
paramset = df.drop(['TARGET_5Yrs','Name'],axis=1).columns.values
df_vals = df.drop(['TARGET_5Yrs','Name'],axis=1).values

paramset_no_3p = df.drop(['TARGET_5Yrs','Name', '3P Made', '3PA', '3P%'],axis=1).columns.values
df_vals_no_3p = df.drop(['TARGET_5Yrs','Name', '3P Made', '3PA', '3P%'],axis=1).values

# replacing Nan values (only present when no 3 points attempts have been performed by a player)
for x in np.argwhere(np.isnan(df_vals)):
    df_vals[x]=0.0

# normalize dataset
X_NO_PROPORTION = MinMaxScaler().fit_transform(df_vals)

X_NO_3P_NO_PROPORTION = MinMaxScaler().fit_transform(df_vals_no_3p)

sm = SMOTE(random_state=42)
X, _ = sm.fit_resample(X_NO_PROPORTION, labels_no_proportion)

sm2 = SMOTE(random_state=42)
X_NO_3P, labels = sm.fit_resample(X_NO_3P_NO_PROPORTION, labels_no_proportion)

# Définition du modèle
ranfom_forest_classifier = RandomForestClassifier(max_depth=20, min_samples_leaf=3, min_samples_split=7, n_estimators=180)

# Le modèle optimisé pour le dataset ne contenant pas les paramètres liés au 3 points permet d'obtenir la valeur de recall la plus élevée
ranfom_forest_classifier_no_3p = RandomForestClassifier(max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=147)

#example of scoring with support vector classifier
score_classifier(X,ranfom_forest_classifier,labels)
score_classifier(X_NO_3P,ranfom_forest_classifier_no_3p,labels)

# Le test de la proportion de chaque classe dans le dataset, le modèle et les valeurs des hyperparamètres ont été choisi dans le fichier test.ipynb

# TODO build a training set and choose a classifier which maximize recall score returned by the score_classifier function


