import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import warnings
warnings.filterwarnings("ignore")


def load_features(file_path):
    """Simple feature loading"""
    print("Loading data...")
    df = pd.read_csv(file_path)

    features_list = []
    labels = []

    for _, row in df.iterrows():
        try:
            # Extract IQ data as integers
            master_ints = [int(x) for x in row["IQ_master"].split()]
            worker_ints = [int(x) for x in row["IQ_worker"].split()]

            # Extract other features
            master_rssi = float(row["master_rssi"])
            worker_rssi = float(row["worker_rssi"])
            aoa_value = float(row["aoa"])
            label = int(row["label"])

            # Combine all features
            features = [master_rssi, worker_rssi, aoa_value] + master_ints + worker_ints

            features_list.append(features)
            labels.append(label)

        except Exception as e:
            print(f"Skipping row due to error: {e}")
            continue

    return np.array(features_list), np.array(labels)


def plot_confusion_matrices(y_true, y_pred, model_name, class_labels):
    """Plots raw and normalized confusion matrices in separate windows"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.round(cm_normalized, 2)
    disp_norm = ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized, display_labels=class_labels
    )
    disp_norm.plot(cmap=plt.cm.Blues)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    # Load data
    file_path = "C:\\Users\\memoo\\esp\\Project\\Training.csv"
    print("Loading data...")
    X, y = load_features(file_path)

    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")

    # Split data (70% train, 10% validation, 20% test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.667, random_state=42, stratify=y_temp
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            max_depth=20,
            min_samples_leaf=10,
            max_features=0.2,
            n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=3, weights="distance", p=2),
        "SVM": SVC(
            C=10.0,
            gamma="scale",
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Train model
        model.fit(X_train_scaled, y_train)

        # Validate model
        y_val_pred = model.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, y_val_pred)

        # Test model
        y_test_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        results[name] = {
            "model": model,
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "test_predictions": y_test_pred,
        }

        print(f"{name} Validation Accuracy: {val_accuracy:.4f}")
        print(f"{name} Test Accuracy: {test_accuracy:.4f}")

        # Plot confusion matrix for current model
        class_labels = np.unique(y)  # Assume integer labels
        plot_confusion_matrices(y_test, y_test_pred, name, class_labels)

    # Summary
    best_model_name = max(results, key=lambda x: results[x]["val_accuracy"])
    best_model = results[best_model_name]["model"]
    best_val_accuracy = results[best_model_name]["val_accuracy"]
    best_test_accuracy = results[best_model_name]["test_accuracy"]

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<15} {'Val Accuracy':<15} {'Test Accuracy':<15}")
    print("-" * 45)

    for name, result in results.items():
        print(
            f"{name:<15} {result['val_accuracy']:<15.4f} {result['test_accuracy']:<15.4f}"
        )

    print(f"\nBest Model: {best_model_name}")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
    print(f"Best Test Accuracy: {best_test_accuracy:.4f}")

    print(f"\nDetailed results for {best_model_name}:")
    print(
        classification_report(
            y_test, results[best_model_name]["test_predictions"]
        )
    )

    # Optional: Save model and scaler
    # joblib.dump(best_model, 'best_model.pkl')
    # joblib.dump(scaler, 'scaler.pkl')
