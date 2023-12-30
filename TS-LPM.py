import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from tqdm import tqdm
from scipy.signal import resample
from scipy.io import loadmat
import random
np.random.seed(42)
random. seed(42)
torch.manual_seed(42)
df = pd.concat([
    pd.read_csv(filepath) for filepath in [
        "/home/xrl/speech/emovo/emovo.csv"
    ]
])

#TARGET_LENGTH = 100
mfcc_features = [scipy.io.loadmat(path)["mfcc"] for path in tqdm(df['path_enmfcc'], desc='Processing MFCC features')]
hubert_features = [scipy.io.loadmat(path)["hubert"] for path in tqdm(df['path_hubert'], desc='Processing hubert features')]
HUBERT_DIM = 768 
MFCC_DIM = 13   
class AttentionFusion(nn.Module):
    def __init__(self, hubert_dim, mfcc_dim):
        super(AttentionFusion, self).__init__()
        self.hubert_transform = nn.Linear(hubert_dim, 512)
        self.mfcc_transform = nn.Linear(mfcc_dim, 512)
        self.attention = nn.Linear(512 * 2, 1)

    def forward(self, hubert, mfcc):
        hubert_transformed = self.hubert_transform(hubert)
        mfcc_transformed = self.mfcc_transform(mfcc)
        concat_features = torch.cat([hubert_transformed, mfcc_transformed], dim=1)

        # Use sigmoid on attention layer output for attention_weights
        attention_weights = torch.sigmoid(self.attention(concat_features))

        fused_features = attention_weights * hubert_transformed + (1 - attention_weights) * mfcc_transformed
        return fused_features

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0001):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.delta = delta
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop
class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(EmotionClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
def normalize_data(data):
    mean = torch.mean(data, dim=0)
    std = torch.std(data, dim=0)
    return (data - mean) / (std + 1e-7)

fusion_model = AttentionFusion(HUBERT_DIM, MFCC_DIM).float()
label_encoder = LabelEncoder()
encoded_emotions = label_encoder.fit_transform(df['emotion'])

class_counts = np.bincount(encoded_emotions)
max_count = np.max(class_counts)
class_weights = max_count / class_counts
weights = torch.tensor(class_weights, dtype=torch.float32)

hubert_mean_tensors = [torch.mean(torch.tensor(feat, dtype=torch.float32), dim=0) for feat in hubert_features]
hubert = normalize_data(torch.stack(hubert_mean_tensors))
print("hubert", hubert.shape)

mfcc_tensors = torch.stack([torch.tensor(feat, dtype=torch.float32) for feat in mfcc_features])
mfcc = normalize_data(torch.mean(mfcc_tensors,dim=1))
print("mfcc",mfcc.shape)

num_emotions = len(np.unique(encoded_emotions))
model = EmotionClassifier(512, num_emotions).float()
criterion = nn.CrossEntropyLoss(weight=weights)
from torch.optim import SGD
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
split = int(0.8 * len(hubert))  # 80% for training
train_data = TensorDataset(fusion_model(hubert[:split], mfcc[:split]), torch.tensor(encoded_emotions[:split]))
test_data = TensorDataset(fusion_model(hubert[split:], mfcc[split:]), torch.tensor(encoded_emotions[split:]))

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)
early_stopping = EarlyStopping(patience=30)
for epoch in range(200):  # Train for 10 epochs
    for fused_features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(fused_features)

        loss = criterion(outputs, labels)
        loss.backward(retain_graph=True)

        optimizer.step()
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for fused_features, labels in test_loader:  # using test_loader as a stand-in for validation
            outputs = model(fused_features)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(test_loader)

    # Print the validation loss for this epoch
    print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}")

    # Step the scheduler
    scheduler.step(avg_val_loss)
    if early_stopping(avg_val_loss):
        print("Early stopping triggered.")
        break
true_labels = []
predictions = []

for fused_features, labels in test_loader:
    with torch.no_grad():
        outputs = model(fused_features)
    _, predicted = outputs.max(1)
    true_labels.extend(labels.tolist())
    predictions.extend(predicted.tolist())

ua = np.mean([accuracy_score(np.array(true_labels) == emotion, np.array(predictions) == emotion) for emotion in np.unique(true_labels)])
print("Unweighted Accuracy (UA):", ua)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(true_labels, predictions)
labels = label_encoder.classes_
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
labels = label_encoder.classes_

plt.figure(figsize=(10, 7))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('/home/xrl/speech/attention_hubert/figure/matrix_emovo.png')
plt.show()
