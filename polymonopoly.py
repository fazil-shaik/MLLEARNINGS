from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)


from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(max_depth=2, random_state=0)
dtree.fit(X_train, y_train)
dtree_preds = dtree.predict(X_test)
dtree_acc = accuracy_score(y_test, dtree_preds)
dtree_cm = confusion_matrix(y_test, dtree_preds)

print("Decision Tree Accuracy:", dtree_acc)

plt.figure(figsize=(4, 3))
sns.heatmap(dtree_cm, annot=True, cmap="Blues", fmt="d")
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


"""
Multi-Level Classification - Real World Examples in Python
Types: Binary, Multi-class, and Multi-label classification
"""

import numpy as np
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

print("=" * 80)
print("MULTI-LEVEL CLASSIFICATION - REAL WORLD EXAMPLES")
print("=" * 80)

# TYPE 1: BINARY CLASSIFICATION
# Real-World Example: Email Spam Detection
print("\n" + "=" * 80)
print("1. BINARY CLASSIFICATION: Email Spam Detection")
print("=" * 80)

# Simulated email data
spam_emails = [
    "Click here to win FREE cash now!!!",
    "Congratulations! You are the lucky winner",
    "Limited time offer - Buy now",
    "URGENT: Claim your prize",
    "Get rich quick scheme",
]

ham_emails = [
    "Meeting scheduled for tomorrow at 2pm",
    "Please find the attached project report",
    "Thank you for your email",
    "Can we reschedule the call?",
    "Your order has been confirmed",
]

emails = spam_emails + ham_emails
labels = [1] * len(spam_emails) + [0] * len(ham_emails)  # 1=Spam, 0=Ham

vectorizer = TfidfVectorizer()
X_email = vectorizer.fit_transform(emails)

X_train_email, X_test_email, y_train_email, y_test_email = train_test_split(
    X_email, labels, test_size=0.3, random_state=42
)

clf_spam = MultinomialNB()
clf_spam.fit(X_train_email, y_train_email)

y_pred_spam = clf_spam.predict(X_test_email)
print(f"\nAccuracy: {accuracy_score(y_test_email, y_pred_spam):.2f}")
print(f"Predictions (0=Ham, 1=Spam): {y_pred_spam}")

# TYPE 2: MULTI-CLASS CLASSIFICATION
# Real-World Example: Iris Flower Classification
print("\n" + "=" * 80)
print("2. MULTI-CLASS CLASSIFICATION: Iris Flower Species")
print("=" * 80)

iris = load_iris()
X_iris = iris.data
y_iris = iris.target
target_names = iris.target_names  # setosa, versicolor, virginica

X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42
)

clf_iris = RandomForestClassifier(n_estimators=100, random_state=42)
clf_iris.fit(X_train_iris, y_train_iris)

y_pred_iris = clf_iris.predict(X_test_iris)
print(f"\nAccuracy: {accuracy_score(y_test_iris, y_pred_iris):.2f}")
print(f"\nClassification Report:")
print(
    classification_report(y_test_iris, y_pred_iris, target_names=target_names)
)

# TYPE 3: MULTI-LABEL CLASSIFICATION
# Real-World Example: Movie Genre Classification
print("\n" + "=" * 80)
print("3. MULTI-LABEL CLASSIFICATION: Movie Genre Tagging")
print("=" * 80)

# Movie descriptions and genres
movies = [
    "A group of thieves plan a heist",  # Action, Crime
    "A love story between two people",  # Romance, Drama
    "Aliens invade Earth",  # Action, Sci-Fi
    "A thriller about a hidden treasure",  # Adventure, Thriller
    "Two people fall in love during war",  # Romance, Drama, War
    "Action-packed spy adventure",  # Action, Adventure, Spy
]

# Multi-label: each movie can have multiple genres
genres = [
    ["Action", "Crime"],
    ["Romance", "Drama"],
    ["Action", "Sci-Fi"],
    ["Adventure", "Thriller"],
    ["Romance", "Drama", "War"],
    ["Action", "Adventure", "Spy"],
]

# Transform labels to binary matrix
mlb = MultiLabelBinarizer()
y_genres = mlb.fit_transform(genres)

vectorizer_movies = TfidfVectorizer(max_features=20)
X_movies = vectorizer_movies.fit_transform(movies)

# Train classifier for each genre independently
from sklearn.multiclass import OneVsRestClassifier

clf_multilabel = OneVsRestClassifier(MultinomialNB())
clf_multilabel.fit(X_movies, y_genres)

# Predict for new movie descriptions
test_movies = [
    "Romantic action adventure in space",
]

X_test_movies = vectorizer_movies.transform(test_movies)
y_pred_genres = clf_multilabel.predict(X_test_movies)

print(f"\nTest movie: '{test_movies[0]}'")
print(f"Predicted genres: {mlb.inverse_transform(y_pred_genres)}")

# BONUS: HIERARCHICAL/MULTI-LEVEL CLASSIFICATION
# Real-World Example: E-commerce Product Classification
print("\n" + "=" * 80)
print("4. HIERARCHICAL MULTI-LEVEL: E-commerce Product Classification")
print("=" * 80)

products = [
    {"name": "iPhone 15", "category": "Electronics", "subcategory": "Phones", "type": "Smartphone"},
    {"name": "Nike Shoes", "category": "Fashion", "subcategory": "Footwear", "type": "Sports"},
    {"name": "Harry Potter Book", "category": "Books", "subcategory": "Fiction", "type": "Fantasy"},
    {"name": "Samsung TV", "category": "Electronics", "subcategory": "TVs", "type": "OLED"},
    {"name": "Jean Pants", "category": "Fashion", "subcategory": "Bottoms", "type": "Casual"},
    {"name": "Python Tutorial", "category": "Books", "subcategory": "Non-Fiction", "type": "Programming"},
]

print("\nProduct Hierarchy:")
for product in products:
    print(
        f"  {product['name']:20} → {product['category']:15} → {product['subcategory']:15} → {product['type']}"
    )

print("\nHierarchy Levels:")
print("  Level 1 (Category):", set(p["category"] for p in products))
print("  Level 2 (Subcategory):", set(p["subcategory"] for p in products))
print("  Level 3 (Type):", set(p["type"] for p in products))

# COMPARISON TABLE
print("\n" + "=" * 80)
print("COMPARISON TABLE: Classification Types")
print("=" * 80)

comparison = """
┌─────────────────────┬──────────────────┬─────────────────────┬──────────────────┐
│ Type                │ # of Classes     │ # of Labels/Sample  │ Real-World Use   │
├─────────────────────┼──────────────────┼─────────────────────┼──────────────────┤
│ Binary              │ 2 (Yes/No)       │ 1 label per sample  │ Spam detection   │
│ Multi-class         │ 3+ (mutually     │ 1 label per sample  │ Iris species     │
│                     │ exclusive)       │                     │                  │
│ Multi-label         │ 3+ (non-         │ Multiple labels per │ Movie genres     │
│                     │ exclusive)       │ sample              │                  │
│ Hierarchical/       │ Organized in     │ Labels at multiple  │ E-commerce       │
│ Multi-level         │ hierarchy        │ levels              │ product taxonomy │
└─────────────────────┴──────────────────┴─────────────────────┴──────────────────┘
"""
print(comparison)

print("\n" + "=" * 80)
print("END OF EXAMPLES")
print("=" * 80)

