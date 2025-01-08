import logging
import os
import sys
import time
import os.path as osp
import torch
from Libs.Datasets.load_dataset import load_UPFD
from Libs.Exp.config import search_args
from Libs.Exp.trainer import infer, init_device, init_wandb, train
from Libs.Models.model4search import Architect, Regressor
from Libs.Utils.logger import init_logger
from Libs.Utils.utils import count_parameters_in_MB, save
from torch.nn import MSELoss
from tqdm import tqdm
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(wandb_runner):
    search_args.save = f'Output/Logs/Search-{search_args.save}'
    log_filename = os.path.join(search_args.save, 'log.txt')

    init_logger('', log_filename, logging.INFO, False)
    print(f'*************log_filename={log_filename}************')

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    logging.info("search_args = %s", search_args.__dict__)

    # dataset_path = osp.join(search_args.dataset_root_path, search_args.dataset_name)

    train_loader, val_loader, test_loader, n_features, n_edge_weight, n_hinfo_node = load_UPFD()
    d_hidden = 128
    d_out = 1
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    model = Classifier(
        criterion,
        d_in,
        d_hinfo_node,
        d_edge_weight,
        d_out,
        d_hidden,
        num_layers=search_args.num_layers,
        epsilon=search_args.epsilon,
        with_conv_linear=search_args.with_conv_linear,
        use_h_info=search_args.use_h_info,
        use_diff_g=search_args.use_diff_g,
        args=search_args
    )
    model = model.to(device)

    logging.info("param size = %fMB", count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        search_args.learning_rate,
        momentum=search_args.momentum,
        weight_decay=search_args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(search_args.trials), eta_min=search_args.learning_rate_min)

    # send model to compute validation loss
    architect = Architect(model, search_args)
    search_cost = 0
    best_res = 0.0
    current_best_genotype = ''
    for trial in (pbar := tqdm(range(search_args.trials))):
        t1 = time.time()
        lr = scheduler.get_last_lr()[0]
        if trial % 1 == 0:
            logging.info('epoch %d lr %e', trial, lr)
            genotype = model.genotype()
            logging.info('genotype = %s', genotype)

        train_obj = train(
            train_loader, model, architect, criterion, optimizer, lr)
        scheduler.step()
        t2 = time.time()
        search_cost += (t2 - t1)

        valid_obj, _, _ = infer(val_loader, model, criterion)
        test_obj, test_msle, test_pcc = infer(
            test_loader, model, criterion, test=True)

        if test_msle > best_res:
            best_res = test_msle
            current_best_genotype = model.genotype()

        if trial % 1 == 0:
            msg = f"[Trial {trial+1}/{search_args.trials}] | Train Loss: {train_obj:.04f} | Val Loss: {valid_obj:.04f} | Test MSLE: {test_msle:.04f}"
            logging.info(msg)
            pbar.set_description(msg)

        save(model, osp.join(search_args.save, 'weights.pt'))
        with open(osp.join(search_args.save, 'best_arch.txt'), 'w') as arch_file:
            arch_file.write(current_best_genotype)

    logging.info('The search process costs %.2fs', search_cost)
    return genotype


def run(random_seed=3407):
    init_device(random_seed)

    # wandb_runner = init_wandb()
    wandb_runner = None
    main(wandb_runner)


run()
