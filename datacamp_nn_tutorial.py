import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import itertools

# Convert data to torch tensors
class Data(Dataset):
    def __init__(self, X, y):
        self.X = X.float()
        self.y = y.float()
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len

# Neural Network Class
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, output_dim)
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.sigmoid(self.layer_2(x))

        return x
       

def main():
    X = torch.rand(10000, 2)  # 10,000 samples with 2 features
    y = torch.randint(0, 2, (10000,))  # 10,000 labels (binary classification)

    # Parameters
    test_size = 0.33
    random_state = 26

    # Set random seed for reproducibility
    torch.manual_seed(random_state)

    # Shuffle the data
    indices = torch.randperm(len(X))
    split_index = int((1 - test_size) * len(X))

    # Split the data
    train_idx, test_idx = indices[:split_index], indices[split_index:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]


    batch_size = 64

    # Instantiate training and test data
    train_data = Data(X_train, y_train)
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    test_data = Data(X_test, y_test)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    # Check 
    '''
    for batch, (X,y) in enumerate(test_dataloader):
        print(f"Batch: {batch + 1}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        break
    '''

    input_dim = 2
    hidden_dim = 10
    output_dim = 1

    model = NeuralNetwork(input_dim, hidden_dim, output_dim)
    #print(model)

    learning_rate = 0.1

    loss_fn = nn.BCELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    num_epochs = 100
    loss_values = []


    for epoch in range(num_epochs):
        for X, y in train_dataloader:
            # zero the parameter gradients
            optimizer.zero_grad()
       
            # forward + backward + optimize
            pred = model(X)
            loss = loss_fn(pred, y.unsqueeze(-1))
            loss_values.append(loss.item())
            loss.backward()
            optimizer.step()
    print ("Training Complete: \n")


    step = torch.linspace(0, 100, steps=10500)
    loss_values = torch.rand(10500)  # Example loss values

    average_loss = loss_values.mean()

    # Print the average loss value
    print(f"Average Loss: {average_loss.item()}")

    y_pred = torch.tensor([], dtype=torch.int64)

    with torch.no_grad():
        for X, y in test_dataloader:
            outputs = model(X)
            predicted = torch.where(outputs < 0.5, torch.tensor(0), torch.tensor(1))
            predicted = list(itertools.chain(*predicted.tolist()))
            y_pred = torch.cat((y_pred,torch.tensor(predicted)), dim=0)
            y_test = torch.cat((y_test,y), dim=0)
            total = y.size(0)
            correct = (torch.tensor(predicted) == y).sum().item()
    print(f'Accuracy of the network on the 3300 test instances: {100 * correct // total}%')




    
    
if __name__ == "__main__":
    main()