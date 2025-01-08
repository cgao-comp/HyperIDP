import datetime
import logging
import os.path as osp
import warnings

import ray
import torch
import torch.nn as nn
from Libs.Data.dataset import get_dataloader
from Libs.Exp.config import tune_args
from Libs.Exp.trainer import infer, train4tune
from Libs.Models.model4tune import Regressor4Tune
from ray import air, tune
from ray.air import session
from ray.tune import ResultGrid
from ray.tune.schedulers import ASHAScheduler
from torch.nn import MSELoss

warnings.simplefilter(action='ignore', category=UserWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_fast_dev(epochs=5):
    with open(tune_args.arch_filename, 'r') as arch_file:
        genotype = arch_file.read()
    train_loader, val_loader, test_loader, n_features, n_edge_weight, n_hinfo_node = get_dataloader()

    d_in = n_features
    d_hidden = tune_args.hidden_size
    d_out = 1

    criterion = MSELoss()
    criterion = criterion.to(device)

    model = Regressor4Tune(
        genotype=genotype,
        criterion=criterion,
        in_dim=d_in,
        out_dim=d_out,
        hidden_size=d_hidden,
        # hidden_size=config['hidden_size'],
        num_layers=tune_args.num_layers,
        in_dropout=tune_args.in_dropout,
        # in_dropout=config['in_dropout'],
        out_dropout=tune_args.out_dropout,
        # out_dropout=config['out_dropout'],
        act=tune_args.activation,
        # act=config['activation'],
        is_mlp=False,
        args=tune_args
    )
    model = model.to(device)

    optimizer = torch.optim.Adagrad(
        model.parameters(),
        tune_args.learning_rate,
        # config['learning_rate'],
        weight_decay=tune_args.weight_decay,
        # config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(tune_args.epochs))
    for epoch in range(epochs):
        train_loss = train4tune(
            train_loader, model, criterion, optimizer)
        if tune_args.cos_lr:
            scheduler.step()

        val_loss, val_msle, _ = infer(
            val_loader, model, criterion)
        test_loss, test_msle, _ = infer(
            val_loader, model, criterion, test=True)

        if epoch % 1 == 0:
            msg = f"[Epoch {epoch+1}/{epochs}] with LR: {scheduler.get_last_lr()[0]:.10f}\n"
            msg += f"<Train> Train Loss: {train_loss:.04f}\n"
            msg += f"<Val> Val Loss: {val_loss:.04f} | Val MSLE: {val_msle:.04f}\n"
            msg += f"<Test> Test MSLE: {test_msle:.04f}"
            print(msg)
            logging.info(msg)


def train4tune_ray(config):
    with open(osp.join(tune_args.root, tune_args.arch_filename), 'r') as arch_file:
        genotype = arch_file.read()
    # hidden_size = tune_args.hidden_size

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    train_loader, val_loader, test_loader, n_features, n_edge_weight, n_hinfo_node = get_dataloader()

    d_in = n_features
    d_out = 1
    model = Regressor4Tune(
        genotype=genotype,
        criterion=criterion,
        in_dim=d_in,
        out_dim=d_out,
        hidden_size=config['hidden_size'],
        num_layers=tune_args.num_layers,
        # in_dropout=tune_args.in_dropout,
        in_dropout=config['in_dropout'],
        # out_dropout=tune_args.out_dropout,
        out_dropout=config['out_dropout'],
        # act=tune_args.activation,
        act=config['activation'],
        is_mlp=False,
        args=tune_args
    )
    model = model.to(device)
    optimizer = config['optimizer'](
        model.parameters(),
        # tune_args.learning_rate,
        config['learning_rate'],
        # weight_decay=tune_args.weight_decay,
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(tune_args.epochs))
    best_val_msle = float('inf')
    for _ in range(tune_args.epochs):
        _ = train4tune(
            train_loader, model, criterion, optimizer)
        if tune_args.cos_lr:
            scheduler.step()

        _, val_msle, _ = infer(
            val_loader, model, criterion)

        if val_msle < best_val_msle:
            best_val_msle = val_msle
            torch.save(model, f'Output/Models/model_msle={best_val_msle:3f}.pth')
        # Send the current training result back to Tune
        if isinstance(best_val_msle, torch.Tensor):
            session.report({"msle": best_val_msle.cpu().numpy()})
        else:
            session.report({"msle": best_val_msle})


def finetune(num_samples=100, max_num_epochs=100, gpus_per_trial=1):
    exp_name = f"{tune_args.dataset_name}/trial_log"
    storage_path = osp.join(tune_args.root, "Output/Logs/FineTune")
    trainable = tune.with_resources(
        train4tune_ray, {"cpu": 16, "gpu": gpus_per_trial})
    tuner = tune.Tuner(
        trainable,
        param_space={'model': 'AutoCPP',
                     'hidden_size': tune.choice([32, 64, 128, 256]),
                     'learning_rate': tune.loguniform(1e-4, 1e-1),
                     'weight_decay': tune.loguniform(5e-4, 5e-3),
                     'optimizer': tune.choice([torch.optim.Adagrad, torch.optim.Adam]),
                     'in_dropout': tune.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
                     'out_dropout': tune.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
                     'activation': tune.choice(['relu', 'elu', 'leaky_relu'])
                     },
        run_config=air.RunConfig(
            name=exp_name,
            stop={"training_iteration": max_num_epochs},
            checkpoint_config=air.CheckpointConfig(
                checkpoint_score_attribute="msle",
                num_to_keep=5,
            ),
            storage_path=storage_path,
        ),
        tune_config=tune.TuneConfig(
            scheduler=ASHAScheduler(
                metric='msle',
                mode='min',
                max_t=max_num_epochs,
                grace_period=1,
                reduction_factor=2
            ),
            num_samples=num_samples,
        ),
    )
    result_grid: ResultGrid = tuner.fit()
    result_grid.get_dataframe().to_csv(osp.join(tune_args.root,
                                                "Output/Results/",
                                                tune_args.dataset_name,
                                                "FineTune",
                                                f'result_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'))


def run():
    if ray.is_initialized():
        ray.shutdown()
    ray.init(logging_level=logging.ERROR)
    finetune()


run()
