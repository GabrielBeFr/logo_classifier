import torch
import wandb
from utils import get_config

def train():
    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define loss_function
    
    # define optimizer and learning rate

    # connect to wandb.ai

    # define metrics

    # define summary.txt content

    # define transform functions (normalization)

    # get dataloaders

    # get model

    # training loop



if __name__ == '__main__':
    wandb.init(project="test-classifier")

    config = get_config("config.yaml")

    train()
