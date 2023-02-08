import torch
import torch.nn as nn
import wandb
from utils import get_config, get_labels
import pathlib
from dataset import get_dataloader
import tqdm
import sklearn.metrics
import numpy as np
from datetime import datetime
import json
import os

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def predict(self, x):
        x = self.linear(x)
        return x

def train(run, log_path, size_epoch = 10000, epochs: int=1):
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

    # define proba function
    softmax = nn.Softmax(dim=1)

    # define lists for metrics
    y_train = []
    y_test = []
    y_pred = []
    classes_str, classes_ids = get_labels("datasets/class_infos.jsonl")

    # training loop
    for epoch in range(epochs):
        count = 0
        model.train()
        train_loss = 0.0
        print("Entering training loop")
        for embeddings, labels, ids in tqdm.tqdm(train_loader):
            count += 1
            optimizer.zero_grad()
            outputs = model.predict(embeddings.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            ground_truth = torch.tensor([torch.where(classe==1)[0] for classe in labels])
            y_train += ground_truth
            if count >= size_epoch:
                break


        # Evaluate the model on validation set
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        missed_logos = {}
        with torch.no_grad():
            print("Entering testing loop")
            for embeddings, labels, ids in tqdm.tqdm(val_loader):
                outputs = model.predict(embeddings.to(device))
                loss = criterion(outputs, labels.to(device))
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                ground_truth = torch.tensor([torch.where(classe==1)[0] for classe in labels])
                correct += (predicted == ground_truth.to(device)).sum().item()
                for indice in torch.where((predicted == ground_truth.to(device))==False)[0].tolist():
                    missed_logos[str(ids[indice].item())]=[ground_truth[indice].item(),predicted[indice].item()]
                y_pred += predicted.tolist()
                y_test += ground_truth.tolist()

        # Compute metrics
        report = sklearn.metrics.classification_report(y_test, y_pred, labels=classes_ids, target_names=classes_str)
        f1_micro = sklearn.metrics.f1_score(y_true=y_test, y_pred=y_pred, labels = classes_ids, average="micro")
        f1_macro = sklearn.metrics.f1_score(y_true=y_test, y_pred=y_pred, labels = classes_ids, average="macro")
        f1_classes = sklearn.metrics.f1_score(y_true=y_test, y_pred=y_pred, labels = classes_ids, average=None)
        total_accuracy = 100 * correct / total
        training_loss = train_loss / len(train_loader)
        validation_loss = val_loss / len(val_loader)
        confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
        metrics_dict = {
            "epoch": epoch + 1,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "total_accuracy": total_accuracy,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            }
        
        for i in range(len(f1_classes)):
            metrics_dict["f1 by class/f1_" + str(classes_str[i])] = f1_classes[i]
        wandb.log(metrics_dict)

        print(f"report: {report}")
        print(f"confusion_matrix: {confusion_matrix}")

if __name__ == '__main__':
    run = wandb.init(project="logos-classifier")

    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S").replace("/","_")
    day, hour = dt_string.split(" ")
    
    dict_path = pathlib.Path("manual_logs/"+day)
    if not os.path.exists(dict_path):
        os.mkdir(dict_path)
    log_path = "manual_logs/"+day+"/"+hour+".json"

    config = get_config("config.yaml")
    run = 5
    train(run, log_path, size_epoch=10000, epochs=10)
    run.finish()
