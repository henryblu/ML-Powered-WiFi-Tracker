# Installing all dependencies
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import (train_test_split,
                                     StratifiedKFold, GridSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
warnings.filterwarnings('ignore')

# Parsing


def parse_iq(iq_string):
    try:
        values = list(map(int, iq_string.strip().split()))
        iq_array = np.array(values).reshape(-1, 2)
        return iq_array[:, 0] + 1j * iq_array[:, 1]
    except Exception:
        return None


# Feature Extraction


def extract_features_from_row(row):
    try:
        iq_master = parse_iq(row["IQ_master"])
        iq_worker = parse_iq(row["IQ_worker"])
        if iq_master is None or iq_worker is None:
            return None
        # Feature extraction from IQ data
        amp_master = np.abs(iq_master)
        phase_master = np.angle(iq_master)
        amp_worker = np.abs(iq_worker)
        phase_worker = np.angle(iq_worker)
        phase_diff = np.unwrap(phase_master - phase_worker)
        amp_diff = amp_master - amp_worker
        features = []
        # Statistical features for each signal
        for signal in [amp_master, amp_worker, phase_diff, amp_diff]:
            features.extend([
                np.mean(signal), np.std(signal), np.var(signal),
                np.min(signal), np.max(signal),
                np.ptp(signal),
                np.median(signal),
                np.percentile(signal, 25),
                np.percentile(signal, 75),
                skew(signal), kurtosis(signal)
            ])
        # Correlation features
        corr_amp = np.corrcoef(amp_master, amp_worker)[0, 1]
        corr_phase = np.corrcoef(phase_master, phase_worker)[0, 1]
        features.append(corr_amp if not np.isnan(corr_amp) else 0)
        features.append(corr_phase if not np.isnan(corr_phase) else 0)
        # Spectral features
        for signal in [amp_master, amp_worker]:
            try:
                freqs, psd = welch(signal, nperseg=min(len(signal), 64))
                features.extend([
                    np.mean(psd), np.std(psd), np.max(psd), np.argmax(psd),
                    np.sum(psd[:len(psd)//4]) / np.sum(psd),
                    np.sum(psd[len(psd)//2:]) / np.sum(psd),
                    np.sum(psd[:len(psd)//10]) / np.sum(psd)
                ])
            except Exception:
                features.extend([0]*7)
        # Phase difference features
        features.extend([
            np.mean(np.abs(phase_diff)), np.std(np.abs(phase_diff)),
            np.mean(np.cos(phase_diff)), np.std(np.cos(phase_diff)),
            np.mean(np.sin(phase_diff)), np.std(np.sin(phase_diff))
        ])
        # RSSI and other features
        master_rssi = float(row.get("master_rssi", -100))
        worker_rssi = float(row.get("worker_rssi", -100))
        features.extend([
            master_rssi, worker_rssi,
            master_rssi - worker_rssi,
            float(row.get("seq_ctrl", 0)),
            float(row.get("aoa", 0))
        ])
        return features
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

#Loading data


def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    X = []
    y = []
    for _, row in df.iterrows():
        features = extract_features_from_row(row)
        if features is not None:
            X.append(features[:-1])
            y.append(int(row["label"]))
    return np.array(X), np.array(y)

# Model Training
def train_model(X_train, y_train, X_test, y_test, model, model_name, param_grid=None):
    print(f"\nTraining {model_name}...")

    if param_grid:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        grid = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        print(f"Best params: {grid.best_params_}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    joblib.dump(model, f"{model_name.lower().replace(' ', '_')}_model.pkl")
    return model, acc


# Plot confusion matrix for each model
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


# Main block
if __name__ == "__main__":
    # Load data
    file_path = "C:\\Users\\memoo\\esp\\Project\\Training.csv"
    print("Loading data...")
    X, y = load_data_from_csv(file_path)

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