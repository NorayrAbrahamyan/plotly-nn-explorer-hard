import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class FeedforwardNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 20)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(20, 20)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


def train_model():
    """
    Trains the neural network using the generated 3D dataset.
    This function loads the data, scales it, and splits it into train and 
    validation sets. It then runs the training loop, saves the loss/accuracy 
    history to a JSON file, and saves the final model weights.
    """
    df = pd.read_csv('data/dataset.csv')
    X = df[['x', 'y', 'z']].values.astype(float)
    y = df['label'].values.astype(float).reshape(-1, 1)

    n_epochs = 200
    model = FeedforwardNN() 
    crit = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.03)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_t, dtype=torch.float32)
    y_train = torch.tensor(y_t, dtype=torch.float32)
    X_valid = torch.tensor(X_v, dtype=torch.float32)
    y_valid = torch.tensor(y_v, dtype=torch.float32)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = crit(y_pred, y_train)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        train_acc = ((y_pred >= 0.5) == y_train).float().mean().item()

        model.eval()
        with torch.no_grad(): 
            y_val_pred = model(X_valid)
            val_loss = crit(y_val_pred, y_valid).item()
            val_acc = ((y_val_pred >= 0.5) == y_valid).float().mean().item()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f'epoch {epoch + 1} | '
                f'train-loss: {train_loss:.4f} val_loss: {val_loss:.4f} | '
                f'train_acc: {train_acc:.4f} val_acc: {val_acc:.4f}'
            )

    with open('model/history.json', 'w') as f:
        json.dump(history, f, indent=4)

    torch.save(model.state_dict(), 'model/model.pth')

if __name__ == "__main__":
    train_model()