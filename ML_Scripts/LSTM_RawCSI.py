import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



# ===================== CONFIGURATION =====================
SEQUENCE_LENGTH = 40
OVERLAP = 20
BATCH_SIZE = 64
EPOCHS = 30
HIDDEN_SIZE = 128
NUM_LAYERS = 3
LEARNING_RATE = 0.0005
DROPOUT = 0.5
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ===================== SPLITTING DATASET =====================


def create_stratified_split(X, y, test_size=0.2, val_size=0.1): #for 70/10/20 train/val/test
    """Create stratified train/val/test split"""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# ===================== SEQUENCE HANDLING =====================
def load_sequences(file_path, sequence_length, step):
    df = pd.read_csv(file_path)

    features_list = []
    label_list = []

    print("Extracting features from all rows...")
    for _, row in df.iterrows():
        features = load_raw_features(row)
        if features is not None:
            features_list.append(features)
            label_list.append(int(row["label"]))

    print(f"Successfully extracted {len(features_list)} feature vectors")

    # Grouping data by class
    class_data = {}
    for i, label in enumerate(label_list):
        if label not in class_data:
            class_data[label] = []
        class_data[label].append(features_list[i])

    # Print class distribution
    print("\nClass distribution:")
    for class_label in sorted(class_data.keys()):
        print(f"Class {class_label}: {len(class_data[class_label])} samples")

    # Create sequences within each class separately
    all_sequences = []
    all_labels = []

    print(f"\nCreating sequences with length={sequence_length}, step={step}")
    for class_label in sorted(class_data.keys()):
        class_features = class_data[class_label]
        class_sequences = []

        # Create sequences within this class only
        for start_idx in range(0, len(class_features) - sequence_length + 1, step):
            sequence = class_features[start_idx:start_idx + sequence_length]
            class_sequences.append(sequence)

        # Add to main lists
        all_sequences.extend(class_sequences)
        all_labels.extend([class_label] * len(class_sequences))

        print("Class {class_label}: {len(class_sequences)} sequences created")

    print("\nTotal sequences created: {len(all_sequences)}")
    print("Final sequence distribution:")
    # Print final sequence distribution per class
    from collections import Counter
    label_counts = Counter(all_labels)
    for class_label in sorted(label_counts.keys()):
        print("Class {class_label}: {label_counts[class_label]} sequences")

    return np.array(all_sequences), np.array(all_labels)


def load_raw_features(row):
    """Extracts features from a single row."""
    try:
        master_rssi = float(row["master_rssi"])
        worker_rssi = float(row["worker_rssi"])
        aoa = float(row["aoa"])

        IQ_master = np.array([int(x) for x in row["IQ_master"].split()])
        IQ_worker = np.array([int(x) for x in row["IQ_worker"].split()])

        # Concatenate scalar and IQ features into one vector
        features = np.concatenate([[master_rssi, worker_rssi, aoa], IQ_master, IQ_worker])
        return features
    except Exception as e:
        print(f"Error processing row: {e}")
        return None

# ===================== LSTM MODEL =====================


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                        batch_first=True, bidirectional=True, dropout=dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=8, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)

        # LSTM layers
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Apply batch normalization (reshape for batch norm)
        lstm_out_reshaped = lstm_out.permute(0, 2, 1)  # (batch, features, seq_len)
        lstm_out_bn = self.batch_norm(lstm_out_reshaped)
        lstm_out_bn = lstm_out_bn.permute(0, 2, 1)  # back to (batch, seq_len, features)

        # Multi-head attention
        # MultiheadAttention expects (seq_len, batch, features)
        lstm_out_transposed = lstm_out_bn.permute(1, 0, 2)
        attn_out, _ = self.attention(lstm_out_transposed, lstm_out_transposed, lstm_out_transposed)

        # Take the mean across sequence dimension for classification
        attn_out = attn_out.permute(1, 0, 2)  # back to (batch, seq_len, features)
        pooled_out = torch.mean(attn_out, dim=1)  # (batch, features)

        # Classification
        return self.classifier(pooled_out)

# ===================== MODEL TRAINING =====================
def train_lstm(X, y):
    print("\n" + "="*50)
    print("Starting LSTM Training with Validation")
    print(f"Input shape: {X.shape}")
    print(f"Sequence length: {SEQUENCE_LENGTH}, Features: {X.shape[2]}")
    print(f"Classes: {len(np.unique(y))}")

    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nSequence distribution per class:")
    for class_label, count in zip(unique, counts):
        print(f"Class {class_label}: {count} sequences")
    print("="*50)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    # Stratified split
    X_train, X_val, X_test, y_train, y_val, y_test = create_stratified_split(
        X, y_encoded, test_size=0.2, val_size=0.1
    )

    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # Scale features
    n_features = X.shape[2]
    scaler = RobustScaler()

    # Fit scaler only on training data
    X_train_flat = X_train.reshape(-1, n_features)
    scaler.fit(X_train_flat)

    # Transform all sets
    X_train_scaled = scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

    # Create datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    model = LSTM(
        input_size=n_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT
    ).to(DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Training variables
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []
    patience_counter = 0
    patience = 10

    # Training loop
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(epoch_val_loss)
        val_accuracies.append(val_acc)

        # Learning rate scheduling
        scheduler.step(epoch_val_loss)

        # Early stopping and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "Best_lstm_model_RawCSI.pt")
            patience_counter = 0
            print(f"Epoch {epoch+1}: New best validation accuracy: {val_acc:.4f}")
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Load best model for final evaluation
    model.load_state_dict(torch.load("best_lstm_model.pt"))

    # Final test evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    final_test_acc = test_correct / test_total
    print(f"\nFinal Test Accuracy: {final_test_acc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.round(cm_normalized, 2)
    # Plot normalized confusion matrix
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=le.classes_)
    disp_norm.plot(cmap=plt.cm.Blues)
    #plt.title("Confusion Matrix - Normalized")
    plt.savefig('LSTM_Raw.png', dpi=300)
    plt.show()

    return model
# Check sequence similarity
def check_sequence_similarity(X, y):
    """Check if sequences are too similar"""

    # Flatten sequences for similarity calculation
    X_flat = X.reshape(X.shape[0], -1)

    # Calculate similarity matrix for first 100 sequences
    if len(X_flat) > 100:
        similarity_matrix = cosine_similarity(X_flat[:100])
        avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
        print(f"Average cosine similarity between sequences: {avg_similarity:.4f}")

        if avg_similarity > 0.8:
            print("⚠️  WARNING: Sequences are highly similar - potential data leakage!")

# ===================== MAIN EXECUTION =====================
if __name__ == "__main__":
    # Set random seeds for reproducibility
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    file_path = "C:\\Users\\memoo\\esp\\Project\\Fine_codes_1\\Training.csv"

    print("Loading and preprocessing data ...")
    print(f"Configuration: SEQ_LEN={SEQUENCE_LENGTH}, OVERLAP={OVERLAP}, BATCH_SIZE={BATCH_SIZE}")

    X_seq, y_seq = load_sequences(file_path, SEQUENCE_LENGTH, SEQUENCE_LENGTH - OVERLAP)
    print(f"Final dataset shape: {X_seq.shape}, Labels shape: {y_seq.shape}")

    # Check for potential data leakage
    print("\nChecking for data quality issues...")
    check_sequence_similarity(X_seq, y_seq)

    # Check class balance
    unique, counts = np.unique(y_seq, return_counts=True)
    print("\nClass distribution in sequences:")
    for class_label, count in zip(unique, counts):
        print(f"  Class {class_label}: {count} sequences ({count/len(y_seq)*100:.1f}%)")

    # Calculate expected sequences per class
    samples_per_class = 1500
    expected_sequences = (samples_per_class - SEQUENCE_LENGTH + 1) // (SEQUENCE_LENGTH - OVERLAP)
    print(f"\nExpected sequences per class: ~{expected_sequences}")
    print(f"Actual total sequences: {len(y_seq)}")

    # Warning if too few sequences
    if len(y_seq) < 500:
        print("WARNING: Very few sequences generated!")
        print("Consider reducing SEQUENCE_LENGTH or OVERLAP for more sequences.")

    # Train improved LSTM model
    print("\n" + "="*60)
    print("TRAINING IMPROVED LSTM MODEL")
    print("="*60)

    # Use improved training function
    lstm_model = train_lstm(X_seq, y_seq)

    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)

    # Additional analysis
    print("\nModel Architecture Summary:")
    total_params = sum(p.numel() for p in lstm_model.parameters())
    trainable_params = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Save final artifacts
    print("\nSaving model artifacts...")
    torch.save(lstm_model.state_dict(), "LSTM_RawCSI.pt")
    print("Model saved successfully!")