from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

emails = [
    "win free money now",
    "claim your free prize",
    "meeting at 10am tomorrow",
    "project report attached",
]
labels = ["spam", "spam", "not spam", "not spam"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)   # turn text into numbers

model = MultinomialNB()
model.fit(X, labels)                   # the model LEARNS here

new_email = ["lunch meeting tomorrow"]
X_new = vectorizer.transform(new_email)

print(model.predict(X_new))            # prediction: ['spam']