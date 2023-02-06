import torch
import torch.nn as nn
import wandb
from utils import get_config
import pathlib
from dataset import get_dataloader
import tqdm


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        x = self.linear(x)
        return x

def train(epochs: int=1):
    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define transform functions (normalization)

    # get dataloaders
    train_path = pathlib.Path("datasets/train_dataset.hdf5")
    val_path = pathlib.Path("datasets/val_dataset.hdf5")
    test_path = pathlib.Path("datasets/test_dataset.hdf5")
    train_loader, val_loader, test_loader = get_dataloader(train_path, val_path, test_path, debugging = False)

    # get model
    input_dim = 512 # number of features in the input data
    output_dim = 168 # number of classes in the target
    model = LinearClassifier(input_dim, output_dim)
    model.to(device)

    # define loss_function
    criterion = torch.nn.CrossEntropyLoss()
    
    # define optimizer and learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # define metrics

    # define summary.txt content

    # training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for embeddings, labels, ids in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model.forward(embeddings.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Evaluate the model on validation set
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        classes_metrics = {}
        with torch.no_grad():
            for embeddings, labels, ids in val_loader:
                outputs = model(embeddings.to(device))
                loss = criterion(outputs, labels.to(device))
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == torch.tensor([torch.where(classe==1)[0] for classe in labels]).to(device)).sum().item()
                
        # Print the results for each epoch
        print('Epoch: {}/{}'.format(epoch+1, epochs))
        print('Training Loss: {:.4f}'.format(train_loss / len(train_loader)))
        print('Validation Loss: {:.4f}'.format(val_loss / len(val_loader)))
        print('Accuracy: {:.2f}%'.format(100 * correct / total))

if __name__ == '__main__':
    wandb.init(project="test-classifier")

    config = get_config("config.yaml")

    train(epochs=15)
