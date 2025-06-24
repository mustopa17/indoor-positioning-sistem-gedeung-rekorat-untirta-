import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Load dataset  
df = pd.read_csv("data_tambahan_10_baris_perkelas.csv")  




# Encode label dan MAC
from sklearn.preprocessing import LabelEncoder

# Label encode untuk kolom Ruangan
le_label = LabelEncoder()
df['label_encoded'] = le_label.fit_transform(df['Ruangan'])

# Label encode untuk kolom mac
le_mac = LabelEncoder()
df['mac_encoded'] = le_mac.fit_transform(df['mac'])


# Fitur dan target
X = df[['mac_encoded', 'RSSI', 'Lantai']]
y = df['label_encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model yang ingin diuji
models = {
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC()
}


best_model = None
best_model_name = ""
best_score = 0


# Evaluasi semua model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== {name} ===")
    print(f"Akurasi: {acc:.2f}")
    print(classification_report(
    y_test, y_pred,
    labels=np.unique(y_test),
    target_names=le_label.inverse_transform(np.unique(y_test)),
    zero_division=0
    
))
    warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50% of the number of samples.")
    
    
    # Simpan model terbaik
    if acc > best_score:
        best_score = acc
        best_model = model
        best_model_name = name
    print(f"Model terbaik : {best_model_name} dengan akurasi {best_score:.2f}")

