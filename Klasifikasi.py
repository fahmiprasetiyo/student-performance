import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the dataset
try:
    df = pd.read_csv('student-mat.csv', sep=';')
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# 1. Feature Engineering

# Create the binary target variable 'pass_status'
# define passing as a final grade (G3) of 10 or higher.
df['pass_status'] = (df['G3'] >= 10).astype(int)  # 1 for Pass, 0 for Fail

# Encode the 'higher' variable (yes/no) into numerical format (1/0)
le = LabelEncoder()
df['higher_encoded'] = le.fit_transform(df['higher'])

# 2. Define Features (X) and Target (y)

# We choose 3 independent variables based on the EDA:
# failures: Strong negative correlation with grades.
# goout: Represents social behavior, has a negative correlation.
# higher: Represents motivation, expected to have a strong positive correlation.
features = ['failures', 'goout', 'higher_encoded']
X = df[features]
y = df['pass_status']

# 3. Split Data into Training and Testing Sets 
# 80% for training, 20% for testing. random_state ensures reproducibility.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train the Classification Model
# We will use Logistic Regression, a standard and interpretable classification algorithm.
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate the Model

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi Model: {accuracy:.2f}")

# Generate and print the Classification Report
print("\nLaporan Klasifikasi:")
# target_names helps label the report
print(classification_report(y_test, y_pred, target_names=['Gagal (0)', 'Lulus (1)']))

# Generate and visualize the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Prediksi Gagal', 'Prediksi Lulus'], 
            yticklabels=['Aktual Gagal', 'Aktual Lulus'])
plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('Label Aktual', fontsize=12)
plt.xlabel('Label Prediksi', fontsize=12)
plt.savefig('confusion_matrix.png')
plt.close()


print("\nClassification analysis complete. confusion_matrix.png has been generated.")
