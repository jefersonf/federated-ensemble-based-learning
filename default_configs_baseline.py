from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import linear_model, svm

DEFAULT_ML_MODELS = {
    "NB":GaussianNB,
    "DT":DecisionTreeClassifier,
    "LR":linear_model.LogisticRegression,
    "SVC":svm.SVC,
    "KNN": KNeighborsClassifier,
    "SGD":SGDClassifier,
    "LDA":LDA
}

DEFAULT_ML_PARAMETERS = {
    "NB":None,  
    "DT":{
        'criterion':['gini','entropy'],
        'splitter':['best','random'],
        'max_depth':[5,10,15,30,50,150,340,None],
    },  
    "LR":{
        "C":[0.001,0.01,0.1,1,10,100],
        'max_iter':[1000,2000,3000,4000]
    },
    "SVC": {
        'C':[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1],
         'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
         'kernel':['rbf','linear']
        },
    "KNN":{
        'n_neighbors':[3,5,10,15,20,30,50],
        'metric':['euclidean','manhattan']
    },
    "SGD":{
        'alpha':[0.0001, 0.001,0.01,0.1,1,10,100],
        'max_iter':[1000,2000,3000,4000],
        'loss':['log']
    },
    "LDA":{
        'solver': ['svd','lsqr'],
        'tol':[0.0001, 0.001,0.01,0.1]
    }
}

DEFAULT_COLUMNS= {
    "BINARY":["ModelType", "Precision", "Recall", "F1Score", "AUC"],
    "MULTI":["ModelType","Precision","Recall","Accuracy"]
}

DEFAULT_NN_CONFIGS = {
    # COVID-19 configs
    "A": [180],
    "B": [19,19],
    "C": [8,8],
    "D": [2,2],
    "E": [19],
    "F": [8],
    "G": [2],
    # Shelter dataset configs
    "H": [18, 18],
    "I": [36, 18],
}