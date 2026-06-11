import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df = sns.load_dataset('iris')

print(df.head())


X = df.drop(['species'],axis=1)
y = df['species']


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)



from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=2)

knn_model.fit(X_train,y_train)

acc_score=knn_model.score(X_test,y_test)

print(f"getting accuracy score knn {acc_score}")

from sklearn.svm import SVC


model_svc = SVC(gamma='auto')
model_svc.fit(X_train,y_train)


acc_score_svc = model_svc.score(X_test,y_test)
print(f"getting accuracy score svc {acc_score_svc}")


#grid search cv on svc

from sklearn.model_selection import GridSearchCV

classifier = GridSearchCV((model_svc),
    {
    'C':[1,3,5,9,11,14],
    'kernel':['linear','rbf'],

},cv=5,return_train_score=False)


classifier.fit(X=X,y=y)

# print(classifier.cv_results_)

results = pd.DataFrame(classifier.cv_results_)


print(f"{results}")

print(results[['param_C','param_kernel','mean_test_score']])


#grid search on knn


classifier_knn = GridSearchCV((knn_model),
                              {
                                  'n_neighbors':[1,3,5,7,11,13],
                                  'algorithm':['kd_tree','ball_tree','brute'],
                                  'leaf_size':[1,4,6,10],
                                  'p':[1,3,7,10],
                              })


classifier_knn.fit(X=X,y=y)

results_knn = pd.DataFrame(classifier_knn.cv_results_)


kNN_result = results_knn[['param_algorithm','param_p','mean_test_score']]

print(kNN_result)
