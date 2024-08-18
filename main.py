import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = pd.read_csv('IMDB Dataset.csv')

# Check the first few rows
print(data.head())

# Preprocessing steps
data['review'] = data['review'].str.lower()  # Convert to lowercase

# Simple text preprocessing function
def preprocess_text(text):
    # Remove punctuation and non-alphabetic characters
    text = ''.join(char for char in text if char.isalpha() or char.isspace())
    return text

data['review'] = data['review'].apply(preprocess_text)

# Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['review'])
y = data['sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Distribution of sentiment classes
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='sentiment')
plt.title('Distribution of Sentiment Classes')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Model Performance Metrics
from sklearn.metrics import precision_recall_fscore_support

report = classification_report(y_test, y_pred, output_dict=True)
classes = list(report.keys())[:-3]  # Excluding 'accuracy', 'macro avg', and 'weighted avg'
precision = [report[cls]['precision'] for cls in classes]
recall = [report[cls]['recall'] for cls in classes]
f1 = [report[cls]['f1-score'] for cls in classes]

x = np.arange(len(classes))  # Convert range to numpy array
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 8))
bars1 = ax.bar(x - width, precision, width, label='Precision')
bars2 = ax.bar(x, recall, width, label='Recall')
bars3 = ax.bar(x + width, f1, width, label='F1-Score')

ax.set_xlabel('Classes')
ax.set_ylabel('Scores')
ax.set_title('Performance Metrics by Class')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()

fig.tight_layout()
plt.show()

# Scatter Plot with Dots for Precision, Recall, and F1-Score
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(x, precision, color='blue', label='Precision', s=100)
ax.scatter(x, recall, color='green', label='Recall', s=100)
ax.scatter(x, f1, color='red', label='F1-Score', s=100)

ax.set_xlabel('Classes')
ax.set_ylabel('Scores')
ax.set_title('Scatter Plot of Performance Metrics by Class')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()

plt.show()
