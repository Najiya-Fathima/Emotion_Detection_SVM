# //
import pandas as pd
import nltk
import re
import joblib
from joblib import Parallel, delayed
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# //
# Download necessary NLTK data files
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
     nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4') # Required for WordNetLemmatizer
except LookupError:
     nltk.download('omw-1.4')

# //
# Load dataset
df = pd.read_csv(r'emotions-dataset\emotions.csv')
print(df.head())
print("\nShape of the dataset:", df.shape)
print("Unique labels: ", df['label'].unique())

#Reduce sample size for faster processing
sample_size = 20000
df_sampled = df.sample(n=sample_size, random_state=42)
print(f"Using {sample_size} samples for faster processing")

# //
# Text preprocessing function
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

# //
# Sequential processing
print("\n------ Cleaning text data ------")
df_sampled['cleaned_text'] = df_sampled['text'].apply(preprocess_text)
print("Text cleaning complete.")

# //
# Prepare data for training
X = df_sampled['cleaned_text']
y = df_sampled['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining data size: {len(X_train)}")
print(f"Testing data size: {len(X_test)}")

# //
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"Feature vector shape: {X_train_vec.shape}")

# //
param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # All 4 kernels
    'C': [0.1, 1, 10]  # Reduced from [0.1, 1, 10, 100] to save time
}

print("\n--- Training SVM Models with GridSearchCV ---")
print(f"Testing {len(param_grid['kernel']) * len(param_grid['C'])} combinations...")

#Reduced CV folds
grid_search = GridSearchCV(
    SVC(random_state=42), 
    param_grid, 
    cv=3,
    n_jobs=-1, 
    scoring='accuracy',
    verbose=1
)

import time
start_time = time.time()
grid_search.fit(X_train_vec, y_train)
training_time = time.time() - start_time

print(f"Training completed in {training_time:.2f} seconds")

# //
# Extract and display results efficiently
best_model = grid_search.best_estimator_
best_accuracy = grid_search.best_score_
best_kernel = grid_search.best_params_['kernel']

print("\n" + "="*60)
print("ACCURACY RESULTS FOR ALL KERNELS")
print("="*60)

# Extract results from GridSearchCV (no need to retrain!)
results_df = pd.DataFrame(grid_search.cv_results_)
kernel_results = {}

for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    kernel_mask = results_df['param_kernel'] == kernel
    kernel_data = results_df[kernel_mask]
    best_idx = kernel_data['mean_test_score'].idxmax()
    best_c = kernel_data.loc[best_idx, 'param_C']
    best_cv_score = kernel_data.loc[best_idx, 'mean_test_score']
    kernel_results[kernel] = {'best_C': best_c, 'cv_accuracy': best_cv_score}
    
    print(f"{kernel.upper():>8} Kernel - Best C: {best_c:>6} | CV Accuracy: {best_cv_score:.4f}")

# //
#Only test the best model on test set (not all kernels)
y_pred = best_model.predict(X_test_vec)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\nBEST MODEL PERFORMANCE:")
print(f"Kernel: {best_kernel.upper()}")
print(f"Parameters: {grid_search.best_params_}")
print(f"Cross-Validation Accuracy: {best_accuracy:.4f}")
print(f"Test Set Accuracy: {test_accuracy:.4f}")

# //
#Quick visualization 

kernels = ['LINEAR', 'POLY', 'RBF', 'SIGMOID']
cv_accuracies = [kernel_results[k.lower()]['cv_accuracy'] for k in kernels]

# Simple bar plot
plt.figure(figsize=(10, 6))
bars = plt.bar(kernels, cv_accuracies, color=['lightblue', 'lightgreen', 'lightcoral', 'lightblue'], alpha=0.8)
plt.xlabel('Kernel Type')
plt.ylabel('Cross-Validation Accuracy')
plt.title(f'SVM Kernel Performance Comparison\n(Sample Size: {sample_size})')
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, acc in zip(bars, cv_accuracies):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

# Highlight the best kernel
best_idx = cv_accuracies.index(max(cv_accuracies))
bars[best_idx].set_color('gold')
bars[best_idx].set_linewidth(2)

plt.tight_layout()
plt.show()

# //
# Save the best model and vectorizer
model_filename = 'best_svm_emotion_model.joblib'
vectorizer_filename = 'tfidf_vectorizer.joblib'

print(f"\nSaving the best model ({best_kernel} kernel) to {model_filename}")
joblib.dump(best_model, model_filename)
joblib.dump(vectorizer, vectorizer_filename)
print("Model and vectorizer saved successfully.")

# //
# Classification report for best model
report = classification_report(y_test, y_pred, zero_division=0)
print(f"\nCLASSIFICATION REPORT (Best Model - {best_kernel.upper()}):")
print(report)
# //
