import os
import numpy as np
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy

# Sequence splitting function
def split_sequence(sequence, n_steps_before, n_steps_after):
    X, y = list(), list()
    for i in range(len(sequence)):
        start_ix = max(0, i - n_steps_before)
        end_ix = min(len(sequence), i + 1 + n_steps_after)
        seq_x_before = sequence[start_ix:i]
        seq_x_after = sequence[i+1:end_ix]
        seq_x = np.concatenate((seq_x_before, seq_x_after), axis=0)
        seq_y = sequence[i]
        while len(seq_x) < n_steps_before + n_steps_after:
            seq_x = np.insert(seq_x, 0, -1)
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# Model building function
def build_model(input_shape, no_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = LSTM(100, activation='relu', return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(150, activation='relu', return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(200, activation='relu')(x)
    outputs = Dense(no_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss=categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    return model

# Load and preprocess data
directory = "dogru4"
kelime = []

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        with open(os.path.join(directory, filename)) as f:
            temp = [line.strip() for line in f.readlines()]
            result = [word.split(' ') for word in temp]
            flat_list = [int(item) for sublist in result for item in sublist]
            kelime.extend(flat_list)

kelime = np.array(kelime)
n_steps_before = 8
n_steps_after = 7
X, y = split_sequence(kelime, n_steps_before, n_steps_after)

X = X.reshape((X.shape[0], X.shape[1], 1)).astype(np.float64)
y = y.astype(np.float64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

no_classes = y_train.shape[1]

# Multiple training runs
n_runs = 10
all_f1_scores = []
all_accuracies = []
all_precisions = []
all_recalls = []
output_file = "metrics_runs_summary.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for run in range(n_runs):
        print(f"\n==== Training Run {run + 1}/{n_runs} ====")
        model = build_model((n_steps_before + n_steps_after, 1), no_classes)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=64,
            verbose=0
        )

        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
        precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
        acc = history.history['val_accuracy'][-1]

        all_f1_scores.append(f1)
        all_accuracies.append(acc)
        all_precisions.append(precision)
        all_recalls.append(recall)

        print(f"Run {run + 1} - F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Val Accuracy: {acc:.4f}")

        f.write(f"Run {run + 1} - F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Val Accuracy: {acc:.4f}\n")

    # Compute means and standard deviations
    mean_f1 = np.mean(all_f1_scores)
    std_f1 = np.std(all_f1_scores, ddof=1)
    mean_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies, ddof=1)
    mean_precision = np.mean(all_precisions)
    std_precision = np.std(all_precisions, ddof=1)
    mean_recall = np.mean(all_recalls)
    std_recall = np.std(all_recalls, ddof=1)

    f.write("\n")
    f.write(f"F1 Score (Mean ± Std): {mean_f1:.4f} ± {std_f1:.4f}\n")
    f.write(f"Precision (Mean ± Std): {mean_precision:.4f} ± {std_precision:.4f}\n")
    f.write(f"Recall (Mean ± Std): {mean_recall:.4f} ± {std_recall:.4f}\n")
    f.write(f"Validation Accuracy (Mean ± Std): {mean_acc:.4f} ± {std_acc:.4f}\n")

print("\nAll runs completed. Metrics saved to:", output_file)
