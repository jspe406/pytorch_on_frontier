import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Generate synthetic data
np.random.seed(0)
X = np.random.rand(1000, 5)  # 100 samples, 2 features
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Binary target

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(5, 10)
        self.output = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

# Initialize the network, loss function, and optimizer
model = SimpleNN()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the network
num_epochs = 10000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the network
with torch.no_grad():
    predicted = model(X_tensor).round()
    accuracy = (predicted.eq(y_tensor).sum() / float(y_tensor.shape[0])).item()
    print(f'Accuracy: {accuracy * 100:.2f}%')
