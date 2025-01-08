import torch
import torch.nn as nn
from Libs.Exp.config import search_args
from torch.autograd import Variable
from torchmetrics import MeanSquaredLogError, PearsonCorrCoef
import os
import logging
import wandb
from torch_geometric import seed_everything
from Libs.Utils.utils import select_gpu
import warnings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_wandb():
    api_key = "a1f83e2d86913e0383d1420133e62af5f0fb0cfd"
    os.environ["WANDB_SILENT"] = "True"
    os.environ["WANDB_NOTEBOOK_NAME"] = "./Notebook.ipynb"
    logger = logging.getLogger("wandb")
    logger.setLevel(logging.ERROR)
    wandb.login(key=api_key,)

    wandb_runner = wandb.init(
        project="AutoCPP",
        notes="Search to Enhance Dynamic GNNs with Information Cascade for Popularity Prediction",
        tags=["baseline", "GNN", "NAS"],
    )
    return wandb_runner


def init_device(seed):
    warnings.filterwarnings('ignore', category=UserWarning)
    select_gpu()
    seed_everything(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def train(loader, model, architect, criterion, optimizer, lr):
    model.train()
    total_loss = 0.0
    # input all data
    architect.step(loader, lr, optimizer, unrolled=search_args.unrolled)
    for batch in loader:
        batch = batch.to(device)
        target = Variable(batch.y).unsqueeze(1).to(device)
        # target = target[:search_args.batch_size, :]
        # train loss
        optimizer.zero_grad()
        output = model(batch).to(device)
        loss = criterion(output, target)

        total_loss += loss.item() / (len(batch)+1)

        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(model.parameters(), search_args.grad_clip)
        optimizer.step()

    return total_loss


def train4tune(loader, model, criterion, optimizer):
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        target = Variable(batch.y).unsqueeze(1).to(device)
        # target = target[:search_args.batch_size, :]
        # train loss
        optimizer.zero_grad()
        output = model(batch).to(device)
        loss = criterion(output, target)

        total_loss += loss.item() / (len(batch)+1)

        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(model.parameters(), search_args.grad_clip)
        optimizer.step()

    return total_loss


def infer(loader, model, criterion, test=False):
    msle, pcc = MeanSquaredLogError().to(device), PearsonCorrCoef().to(device)
    model.eval()
    total_loss = 0.0
    total_msle, total_pcc = 0.0, 0.0
    for batch in loader:
        batch = batch.to(device)
        target = Variable(batch.y).unsqueeze(1).to(device)
        # target = target[:search_args.batch_size, :]
        with torch.no_grad():
            output = model(batch).to(device)
        loss = criterion(output, target)
        total_loss += loss.item() / (len(batch)+1)
        total_msle += msle(output, target) / (len(batch)+1)
        total_pcc += pcc(output, target) / (len(batch)+1)

    return total_loss, total_msle, total_pcc
