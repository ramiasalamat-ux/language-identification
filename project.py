# ===============================
# Language Identification Project
# ===============================

# -------- CELL 1: Install Libraries --------



# -------- CELL 2: Imports --------
from datasets import load_dataset
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import seaborn as sns
import matplotlib.pyplot as plt


# -------- CELL 3: Load Dataset --------
dataset = load_dataset("papluca/language-identification")

train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

print(train_df.head())


# -------- CELL 4: Explore Data --------
print(train_df['labels'].value_counts())


# -------- CELL 5: Character n-grams --------
char_vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(2,5),
    max_features=50000
)

X_train_char = char_vectorizer.fit_transform(train_df['text'])
X_test_char = char_vectorizer.transform(test_df['text'])


# -------- CELL 6: Word n-grams --------
word_vectorizer = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1,2),
    max_features=50000
)

X_train_word = word_vectorizer.fit_transform(train_df['text'])
X_test_word = word_vectorizer.transform(test_df['text'])


# -------- CELL 7: Training Function --------
def train_and_evaluate(model, X_train, X_test, name):
    model.fit(X_train, train_df['labels'])
    preds = model.predict(X_test)
    
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(test_df['labels'], preds))
    print(classification_report(test_df['labels'], preds))
    
    return preds


# -------- CELL 8: Naive Bayes --------
nb = MultinomialNB()
pred_nb = train_and_evaluate(nb, X_train_char, X_test_char, "Naive Bayes (Char)")


# -------- CELL 9: Logistic Regression --------
lr = LogisticRegression(max_iter=200)
pred_lr = train_and_evaluate(lr, X_train_char, X_test_char, "Logistic Regression (Char)")


# -------- CELL 10: Linear SVM --------
svm = LinearSVC()
pred_svm = train_and_evaluate(svm, X_train_char, X_test_char, "Linear SVM (Char)")


# -------- CELL 11: Confusion Matrix --------
cm = confusion_matrix(test_df['labels'], pred_svm)

plt.figure(figsize=(12,10))
sns.heatmap(cm)
plt.title("Confusion Matrix (SVM)")
plt.show()


# -------- CELL 12: Error Analysis --------
errors = test_df.copy()
errors['pred'] = pred_svm

wrong = errors[errors['labels'] != errors['pred']]
print(wrong.head(10))


# -------- CELL 13: Compare Word n-grams (Optional) --------
print("\n--- Using WORD n-grams with SVM ---")
pred_svm_word = train_and_evaluate(LinearSVC(), X_train_word, X_test_word, "SVM (Word)")


# -------- CELL 14: Observations (Write in Report) --------
# 1. Character n-grams perform better than word n-grams
# 2. Similar languages cause confusion (e.g., Spanish vs Portuguese)
# 3. Script-based languages are easier to classify
# 4. Linear SVM gives best performance
