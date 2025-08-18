# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 18:39:34 2023

@author: sena
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, precision_score, recall_score

# Data loading
folder_paths = ["eksik3", "fazla3", "yanlis3"]
texts = []
labels = []

for folder_path in folder_paths:
    folder_label = folder_paths.index(folder_path)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            texts.append(text)
            labels.append(folder_label)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

max_sequence_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

labels = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42, shuffle=True)

# Model building function
def build_model():
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index) + 1, 128, input_length=max_sequence_length))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(256)))
    model.add(Dropout(0.5))
    model.add(Dense(len(folder_paths), activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer=Adam(learning_rate=0.0001), 
                  metrics=['accuracy'])
    return model

# Multiple training runs
n_runs = 10
all_f1_scores = []
all_accuracies = []
all_precisions = []
all_recalls = []
all_train_accuracies = []
output_file = "text_classification_metrics_runs.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for run in range(n_runs):
        print(f"\n==== Training Run {run + 1}/{n_runs} ====")
        model = build_model()
        history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), verbose=0)

        y_pred = model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = y_val

        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
        precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
        val_acc = history.history['val_accuracy'][-1]
        train_acc = history.history['accuracy'][-1]

        all_f1_scores.append(f1)
        all_accuracies.append(val_acc)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_train_accuracies.append(train_acc)

        print(f"Run {run + 1} - Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        f.write(f"Run {run + 1} - Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}\n")

    # Compute means and standard deviations
    mean_f1 = np.mean(all_f1_scores)
    std_f1 = np.std(all_f1_scores, ddof=1)
    mean_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies, ddof=1)
    mean_precision = np.mean(all_precisions)
    std_precision = np.std(all_precisions, ddof=1)
    mean_recall = np.mean(all_recalls)
    std_recall = np.std(all_recalls, ddof=1)
    mean_train_acc = np.mean(all_train_accuracies)
    std_train_acc = np.std(all_train_accuracies, ddof=1)

    f.write("\n")
    f.write(f"Train Accuracy (Mean ± Std): {mean_train_acc:.4f} ± {std_train_acc:.4f}\n")
    f.write(f"Validation Accuracy (Mean ± Std): {mean_acc:.4f} ± {std_acc:.4f}\n")
    f.write(f"F1 Score (Mean ± Std): {mean_f1:.4f} ± {std_f1:.4f}\n")
    f.write(f"Precision (Mean ± Std): {mean_precision:.4f} ± {std_precision:.4f}\n")
    f.write(f"Recall (Mean ± Std): {mean_recall:.4f} ± {std_recall:.4f}\n")

print("\nAll runs completed. Metrics saved to:", output_file)
