import re
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# 1. Load the dataset
data = pd.read_csv("emails.csv")

data = data.dropna(subset=["email", "label"])
data["label"] = data["label"].str.strip().str.title()
data = data[data["label"].isin(["Phishing", "Safe"])]

if data["label"].nunique() != 2:
    raise ValueError("Dataset must contain both Phishing and Safe emails.")

if data["label"].value_counts().min() < 2:
    raise ValueError("Each class needs at least two examples.")

# 2. Extract simple URL and keyword features
suspicious_words = [
    "urgent", "verify", "password", "suspended",
    "blocked", "claim", "prize", "banking",
    "click here", "login", "refund"
]


def prepare_email(email):
    text = str(email).lower()

    urls = re.findall(r"https?://\S+|www\.\S+", text)
    keyword_count = sum(
        text.count(word) for word in suspicious_words
    )

    # Add simple feature indicators to the email text.
    # TF-IDF learns patterns from these indicators too.
    text += f" URL_COUNT_{min(len(urls), 5)}"
    text += f" SUSPICIOUS_WORD_COUNT_{min(keyword_count, 10)}"

    return text


X = data["email"].apply(prepare_email)
y = data["label"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# 4. Create and train the ML model
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

# 5. Evaluate the model
predictions = model.predict(X_test)

print("\n===== PHISHING EMAIL DETECTOR =====")
print("Total emails:", len(data))
print("Training emails:", len(X_train))
print("Testing emails:", len(X_test))

accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=["Phishing", "Safe"],
    zero_division=0
))

# Fixed order: rows = actual, columns = predicted
labels = ["Phishing", "Safe"]
matrix = confusion_matrix(
    y_test, predictions, labels=labels
)

print("Confusion Matrix:")
print(matrix)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=labels
)
display.plot(cmap="Blues")
plt.title("Phishing Email Detection")
plt.tight_layout()
plt.show()

# 6. Test a new email
print("\nEnter an email to analyze.")
new_email = input("Email text: ")

prepared = prepare_email(new_email)
prediction = model.predict([prepared])[0]

print("\nPrediction:", prediction)

if prediction == "Phishing":
    print("Warning: This email may be phishing.")
else:
    print("This email was classified as safe.")

print("Always verify suspicious emails manually.")
