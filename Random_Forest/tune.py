import pandas as pd
from sklearn.model_selection import KFold

from practical2.random_forest import RandomForestClassifier
from practical2.random_forest import f1_score, data_preprocess
        

def tune(X, Y):
    
    n__estimators = np.arange(10, 100, 40)
    split__size = np.arange(0.6, 1, 0.4)
    max__features = np.arange(0.2, 1, 0.4)
    
    N =len(n__estimators)*len(split__size)*len(max__features)
    count = 1
    accuracy = 0
    
    for n_estimators in n__estimators:
        for split_size in split__size:
            for max_features in max__features:
                acc = []
                print(f"[TUNING] {count} out of {N}")
                count = count + 1
                
                kf = KFold(n_splits=10)
                for train_index, test_index in kf.split(X):
    
                    X_train, x_test = X[train_index], X[test_index]
                    Y_train, y_test = Y[train_index], Y[test_index]
                
                    RF = RandomForestClassifier(n_estimators = n_estimators, split_size = split_size, max_features = max_features)
                    RF.fit(X_train, Y_train)
                    
                    acc.append(f1_score(y_test, RF.predict(x_test)))
                    
                if np.mean(acc) > accuracy:
                    accuracy = np.mean(acc)
                    best_n_estimators = n_estimators
                    best_split_size = split_size
                    best_max_features = max_features
                
    print("[NUMBER OF ESTIMATORS]", best_n_estimators)   
    print("[SPLIT SIZE]          ", best_split_size) 
    print("[MAX FEATURES]        ", best_max_features)   
    print("[ACCURACY]            ", accuracy)

model = RandomForestClassifier()

train_data = pd.read_csv("data/train.csv", encoding = "latin1")
df = pd.DataFrame(train_data)

labels = train_data['label'].values
x_train = np.array(train_data.drop('label', axis=1))
y_train = labels

x_train = data_preprocess(x_train)

tune(x_train, y_train)

