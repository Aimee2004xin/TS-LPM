import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import scipy.io
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
# Load the CSV file
df = pd.read_csv("/home/xrl/speech/savee/savee.csv")

import random
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


from scipy.interpolate import interp1d

def resample_feature(file_path, target_length=300):
    try:
        mat = scipy.io.loadmat(file_path)
        wavlm_feature = mat['hubert']

        # Remove the first dimension, which is '1'
        #wavlm_feature = np.squeeze(wavlm_feature, axis=0)

        original_length = wavlm_feature.shape[0]
        resampled_feature = np.zeros((target_length, wavlm_feature.shape[1]))

        for i in range(wavlm_feature.shape[1]):
            y = wavlm_feature[:, i]
            x = np.linspace(0, 1, original_length)
            if len(x) != len(y):
                print(f"Length mismatch in file: {file_path}, feature index: {i}")
                continue
            f = interp1d(x, y, kind='linear')
            x_new = np.linspace(0, 1, target_length)
            resampled_feature[:, i] = f(x_new)

        return resampled_feature

    except Exception as e:
        print(f"Error processing file: {file_path}, Error: {e}")
        return np.zeros((target_length, wavlm_feature.shape[1] if 'wavlm_feature' in locals() else 0))


wavlm_features = []
for path in tqdm(df['path_hubert'], desc='Processing hubert features'):
    wavlm_features.append(resample_feature(path))
hubert_features = np.array(wavlm_features)
# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df['emotion'])

# Define the classifier
class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(EmotionClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# Prepare data for training
# Normalize the features
def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / (std + 1e-7)

normalized_features = normalize_data(np.array(wavlm_features))

# Assuming each resampled feature is of shape (200, feature_size)
# Flatten the features for the neural network
flattened_features = normalized_features.reshape(normalized_features.shape[0], -1)
input_dim = flattened_features.shape[1]  # Update input dimension

# Rest of your code for training and evaluation

num_classes = len(np.unique(labels))
model = EmotionClassifier(input_dim, num_classes)
from torch.optim import Adam
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(flattened_features, labels, test_size=0.2, random_state=42)

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# DataLoader
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# Model, Loss, Optimizer
model = EmotionClassifier(input_dim, num_classes).float()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
#for epoch in range(200):  # Number of epochs
for epoch in tqdm(range(200), desc='Training'):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Testing loop
true_labels = []
predictions = []

model.eval()
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output, 1)
        true_labels.extend(target.tolist())
        predictions.extend(predicted.tolist())

# Evaluation Metrics
wa = accuracy_score(true_labels, predictions)
ua = np.mean([accuracy_score(np.array(true_labels) == emotion, np.array(predictions) == emotion) for emotion in np.unique(true_labels)])
wf1 = f1_score(true_labels, predictions, average='weighted')
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# Confusion Matrix
cm = confusion_matrix(true_labels, predictions)
labels = label_encoder.classes_


# Normalize the confusion matrix so that each row sums to 1
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Get the unique labels from the true_labels
labels = label_encoder.classes_

# Plotting using seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('/home/xrl/speech/hubert/hubertmatrix/matrix_savee.png')
plt.show()
# Print metrics
print("Weighted Accuracy (WA):", wa)
print("Unweighted Accuracy (UA):", ua)
print("Weighted F1 Score (WF1):", wf1)