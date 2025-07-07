import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import warnings
warnings.filterwarnings('ignore')

#COMMON PARSE FUNCTION
def parse_iq(iq_string):
    try:
        values = list(map(int, iq_string.strip().split()))
        iq_array = np.array(values).reshape(-1, 2)
        return iq_array[:, 0] + 1j * iq_array[:, 1]
    except:
        return None

#FEATURE EXTRACTORS
def extract_knn_features(file_path, class_label):
    df = pd.read_csv(file_path)
    X, y = [], []
    for _, row in df.iterrows():
        iq_master = parse_iq(row["IQ_master"])
        iq_worker = parse_iq(row["IQ_worker"])
        if iq_master is None or iq_worker is None:
            continue
        phase_diff = np.angle(iq_master) - np.angle(iq_worker)
        amp_diff = np.abs(iq_master) - np.abs(iq_worker)
        master_real_imag = np.concatenate([iq_master.real, iq_master.imag])
        worker_real_imag = np.concatenate([iq_worker.real, iq_worker.imag])
        extra_features = []
        for col in df.columns:
            if col not in ["IQ_master", "IQ_worker"]:
                try:
                    extra_features.append(float(row[col]))
                except:
                    continue
        features = np.concatenate([
            master_real_imag,
            worker_real_imag,
            phase_diff,
            amp_diff,
            extra_features
        ])
        X.append(features)
        y.append(class_label)
    return X, y

def extract_svm_features(df, class_label):
    X, y = [], []
    for _, row in df.iterrows():
        iq_master = parse_iq(row["IQ_master"])
        iq_worker = parse_iq(row["IQ_worker"])
        if iq_master is None or iq_worker is None:
            continue
        amp_master = np.abs(iq_master)
        phase_master = np.angle(iq_master)
        amp_worker = np.abs(iq_worker)
        phase_worker = np.angle(iq_worker)
        phase_diff = np.unwrap(phase_master - phase_worker)
        amp_diff = amp_master - amp_worker
        amp_ratio = amp_master / (amp_worker + 1e-8)
        features = []
        for signal in [amp_master, amp_worker, phase_diff, amp_diff]:
            features.extend([
                np.mean(signal), np.std(signal), np.var(signal), np.min(signal),
                np.max(signal), np.ptp(signal), np.median(signal),
                np.percentile(signal, 25), np.percentile(signal, 75),
                skew(signal), kurtosis(signal)
            ])
        corr_amp = np.corrcoef(amp_master, amp_worker)[0, 1]
        corr_phase = np.corrcoef(phase_master, phase_worker)[0, 1]
        features.append(corr_amp if not np.isnan(corr_amp) else 0)
        features.append(corr_phase if not np.isnan(corr_phase) else 0)
        for signal in [amp_master, amp_worker]:
            try:
                freqs, psd = welch(signal, nperseg=min(len(signal), 32))
                features.extend([
                    np.mean(psd), np.std(psd), np.max(psd), np.argmax(psd),
                    np.sum(psd[:len(psd)//4]) / np.sum(psd),
                    np.sum(psd[len(psd)//2:]) / np.sum(psd)
                ])
            except:
                features.extend([0, 0, 0, 0, 0, 0])
        features.extend([
            np.mean(np.abs(phase_diff)), np.std(np.abs(phase_diff)),
            np.mean(np.cos(phase_diff)), np.std(np.cos(phase_diff)),
            np.mean(np.sin(phase_diff)), np.std(np.sin(phase_diff)),
            np.mean(amp_ratio), np.std(amp_ratio), np.min(amp_ratio),
            np.max(amp_ratio), np.median(amp_ratio)
        ])
        master_rssi = float(row.get("master_rssi", -100))
        worker_rssi = float(row.get("worker_rssi", -100))
        features.extend([master_rssi, worker_rssi, master_rssi - worker_rssi, master_rssi / (worker_rssi + 1e-8)])
        features.append(float(row.get("seq_ctrl", 0)))
        features.append(float(row.get("aoa", 0)))
        phase_diff_mean = np.mean(phase_diff)
        esp_separation = 0.06
        wavelength = 0.125
        estimated_path_diff = (phase_diff_mean * wavelength) / (2 * np.pi)
        features.append(estimated_path_diff)
        if abs(estimated_path_diff) < esp_separation:
            angle_estimate = np.arcsin(estimated_path_diff / esp_separation) * 180 / np.pi
        else:
            angle_estimate = 90 * np.sign(estimated_path_diff)
        features.append(angle_estimate)
        X.append(features)
        y.append(class_label)
    return X, y

#LOAD DATA
def load_data(folder_path, file_class_map, extractor_func):
    X, y = [], []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            for key, label in file_class_map.items():
                if key in filename:
                    try:
                        df = pd.read_csv(file_path)
                        if extractor_func.__name__ == "extract_knn_features":
                            x, label_list = extractor_func(file_path, label)
                        else:
                            x, label_list = extractor_func(df, label)
                        X.extend(x)
                        y.extend(label_list)
                        print(f"Data Loaded {filename} for class {label}")
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
                    break
    return np.array(X), np.array(y)

#CLASSIFIERS 
def train_knn(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("\n KNN Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    joblib.dump(knn, "knn_model.pkl")
    return knn

def train_svm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    selector = SelectKBest(score_func=f_classif, k=min(50, X_train.shape[1]))
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel = selector.transform(X_test_scaled)
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_sel)
    X_test_pca = pca.transform(X_test_sel)
    param_grid = {
        'C': [1, 10],
        'gamma': ['scale', 0.01],
        'kernel': ['rbf']
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=cv, n_jobs=-1)
    grid.fit(X_train_pca, y_train)
    y_pred = grid.predict(X_test_pca)
    print("\n SVM Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    joblib.dump(grid.best_estimator_, "svm_model.pkl")
    return grid.best_estimator_

# MAIN 
if __name__ == "__main__":
    folder_path = "C:\\Users\\memoo\\esp\\Project\\DS\\"
    file_class_map = {
        "2m-90deg": 0, "2m-45deg": 1, "2m-0deg": 2, "2m--45deg": 3, "2m--90deg": 4,
        "5m-90deg": 5, "5m-45deg": 6, "5m-0deg": 7, "5m--45deg": 8, "5m--90deg": 9
    }

    print("\n Loading KNN data...")
    X_knn, y_knn = load_data(folder_path, file_class_map, extract_knn_features)
    print(f"KNN data: {X_knn.shape}")
    knn_model = train_knn(X_knn, y_knn)

    print("\n Loading SVM data...")
    X_svm, y_svm = load_data(folder_path, file_class_map, extract_svm_features)
    print(f"SVM data: {X_svm.shape}")
    svm_model = train_svm(X_svm, y_svm)
