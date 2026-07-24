from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
from sklearn.preprocessing import LabelEncoder



df = sns.load_dataset('iris')

df.head()


X = df.drop('species', axis=1)
y = df['species']

le=LabelEncoder()
y=le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


base_learners = [
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('lr', LogisticRegression(random_state=42)),
    ('svc', SVC(probability=True, random_state=42))
]


meta_learner = LogisticRegression(random_state=42,max_iter=1000)


check_prediction = base_learners[0][1].fit(X_train, y_train).predict(X_test)

print("Base Learner (Decision Tree) Accuracy:", accuracy_score(y_test, check_prediction))

stacking_clf = StackingClassifier(estimators=base_learners, final_estimator=meta_learner,cv=5)

stacking_clf.fit(X_train, y_train)

y_pred = stacking_clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
