import torch
from torch.utils.tensorboard import SummaryWriter

from data_utils import load_and_preprocess_data
from model import BinaryClassification
from train import train
from evaluate import evaluate

DATA_PATH = r'data\apple_quality.csv'


def main():
    train_loader, test_loader, input_dim = load_and_preprocess_data(DATA_PATH)
    model = BinaryClassification(input_dim)
    optimizer = torch.optim.Adam(model.parameters())
    writer = SummaryWriter()

    train(model, train_loader, optimizer, writer)
    writer.close()

    evaluate(model, test_loader)

    model_name = input('Model name: ')
    torch.save(model.state_dict(), f'./models/{model_name}.pth')


if __name__ == '__main__':
    main()
