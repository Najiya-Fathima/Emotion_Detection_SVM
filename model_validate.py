import joblib

# Loading the saved model and vectorizer
loaded_model = joblib.load('best_svm_emotion_model.joblib')
loaded_vectorizer = joblib.load('tfidf_vectorizer.joblib')

# New text to predict
test_texts = [
    "I'm feeling really happy and excited today!",
    "Why does everything seem so hopeless?",
    "I don't care anymore. Leave me alone.",
    "That's amazing news! I'm thrilled.",
    "I'm just so angry right now I can't think straight.",
    "Honestly, I'm feeling very confused and unsure.",
]

# Preprocess the new text (same preprocessing function)
def preprocess_text(text):

    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    import re
    
    # Initialize lemmatizer and stop words inside the function
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)  # Remove user mentions and hashtags
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    tokens = word_tokenize(text)  # Tokenize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]  # Remove stopwords and lemmatize
    
    return ' '.join(tokens)

preprocessed_texts = [preprocess_text(text) for text in test_texts]

# Transform using the SAME vectorizer that was used during training
X_input = loaded_vectorizer.transform(preprocessed_texts)

# Make prediction
predictions = loaded_model.predict(X_input)

print("\n--- Emotion Predictions on New Texts ---")
for original, pred in zip(test_texts, predictions):
    print(f"Text: {original}\nPredicted Emotion: {pred}\n")
