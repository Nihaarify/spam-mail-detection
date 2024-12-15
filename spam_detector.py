import pandas as pd  # For data handling
import numpy as np   # For numerical operations
import string        # For handling punctuation
import nltk          # For text processing
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer  # Convert text to numbers
from sklearn.model_selection import train_test_split  # Split data
from sklearn.naive_bayes import MultinomialNB  # Model
from sklearn.metrics import accuracy_score, confusion_matrix  # Evaluate model

# Download stopwords for NLTK
nltk.download('stopwords')
# Load the dataset
data = pd.read_csv("spam_data.csv")

# Display the first few rows
print("Dataset loaded successfully:")
print(data.head())
# Encode labels: 'ham' -> 0, 'spam' -> 1
data['Label'] = data['Label'].map({'ham': 0, 'spam': 1})

# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = "".join(char for char in text if char not in string.punctuation)  # Remove punctuation
    words = text.split()  # Tokenize
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return " ".join(words)

# Apply cleaning to the "Text" column
data['Cleaned_Text'] = data['Text'].apply(clean_text)

print("\nCleaned Data:")
print(data[['Text', 'Cleaned_Text']].head())
# Convert text into numerical features
tfidf = TfidfVectorizer(max_features=3000)  # Use top 3000 words
X = tfidf.fit_transform(data['Cleaned_Text']).toarray()  # Convert to array
y = data['Label']  # Labels

print("\nShape of X (features):", X.shape)
print("Shape of y (labels):", y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
# Initialize and train the model
model = MultinomialNB()
model.fit(X_train, y_train)

print("\nModel Training Complete!")
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)