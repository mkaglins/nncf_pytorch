import random
import sys
from pathlib import Path

import copy
import os.path as osp
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.models as models
import warnings
import torch
from shutil import copyfile

from examples.classification.LeGR_DDPG import LeGREnv, RL_agent_train, DDPGAgentOptimizer, LeGRPruner
from examples.classification.main import load_resuming_checkpoint, validate, train, create_datasets, create_data_loaders
from examples.classification.random_agent import run_random_agent
from examples.common.argparser import get_common_argument_parser
from examples.common.distributed import configure_distributed
from examples.common.example_logger import logger
from examples.common.execution import ExecutionMode, get_device, get_execution_mode, \
    prepare_model_for_execution, start_worker
from examples.common.model_loader import load_model
from examples.common.optimizer import get_parameter_groups, make_optimizer
from examples.common.sample_config import SampleConfig, create_sample_config
from examples.common.utils import configure_logging, configure_paths, create_code_snapshot, \
    print_args, print_statistics, \
    is_pretrained_model_requested
from examples.common.utils import write_metrics
from nncf import create_compressed_model, load_state
from RL_agent import run_rl_agent
from nncf.initialization import register_default_init_args
from nncf.utils import manual_seed, is_main_process
import numpy as np

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def get_argument_parser():
    parser = get_common_argument_parser()
    parser.add_argument(
        "--dataset",
        help="Dataset to use.",
        choices=["imagenet", "cifar100", "cifar10"],
        default=None
    )
    parser.add_argument('--test-every-n-epochs', default=1, type=int,
                        help='Enables running validation every given number of epochs')
    return parser


def main(argv):
    parser = get_argument_parser()
    args = parser.parse_args(args=argv)
    config = create_sample_config(args, parser)

    if config.dist_url == "env://":
        config.update_from_env()

    configure_paths(config)
    copyfile(args.config, osp.join(config.log_dir, 'config.json'))
    source_root = Path(__file__).absolute().parents[2]  # nncf root
    create_code_snapshot(source_root, osp.join(config.log_dir, "snapshot.tar.gz"))

    if config.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    config.execution_mode = get_execution_mode(config)

    if config.metrics_dump is not None:
        write_metrics(0, config.metrics_dump)

    start_worker(main_worker, config)


# pylint:disable=too-many-branches
def main_worker(current_gpu, config: SampleConfig):
    config.current_gpu = current_gpu
    config.distributed = config.execution_mode in (ExecutionMode.DISTRIBUTED, ExecutionMode.MULTIPROCESSING_DISTRIBUTED)
    if config.distributed:
        configure_distributed(config)

    config.device = get_device(config)

    if is_main_process():
        configure_logging(logger, config)
        print_args(config)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(config.device)

    train_loader = train_sampler = val_loader = None
    resuming_checkpoint_path = config.resuming_checkpoint_path
    nncf_config = config.nncf_config

    train_dataset, val_dataset = create_datasets(config)
    train_loader, train_sampler, val_loader = create_data_loaders(config, train_dataset, val_dataset)
    pretrained = is_pretrained_model_requested(config)
    nncf_config = register_default_init_args(nncf_config, criterion, train_loader)

    # Data loading code
    train_dataset, val_dataset = create_datasets(config)
    train_loader, train_sampler, val_loader = create_data_loaders(config, train_dataset, val_dataset)

    coeff_range = [0]  # [0.5, 1, 2, 5, 10, 100]
    for LOSS in [False]:
        for coeff in coeff_range:
            # create model
            model_name = config['model']
            model = load_model(model_name,
                               pretrained=pretrained,
                               num_classes=config.get('num_classes', 1000),
                               model_params=config.get('model_params'),
                               weights_path=config.get('weights'))

            model.to(config.device)

            resuming_model_sd = None
            resuming_checkpoint = None
            if resuming_checkpoint_path is not None:
                resuming_checkpoint = load_resuming_checkpoint(resuming_checkpoint_path)
                resuming_model_sd = resuming_checkpoint['state_dict']

            compression_ctrl, model = create_compressed_model(copy.deepcopy(model), nncf_config, resuming_state_dict=resuming_model_sd)

            model, _ = prepare_model_for_execution(model, config)
            if config.distributed:
                compression_ctrl.distributed()

            # define optimizer
            params_to_optimize = get_parameter_groups(model, config)
            optimizer, lr_scheduler = make_optimizer(params_to_optimize, config)

            best_acc1 = 0
            if config.execution_mode != ExecutionMode.CPU_ONLY:
                cudnn.benchmark = True

            # TRY TO RUN RL
            # reward, acc, actions = run_rl_agent(logger, model, compression_ctrl, (train_loader, val_loader), coeff, LOSS)
            while isinstance(model, nn.DataParallel):
                model = model.module
            RL_agent_train(LeGREnv, DDPGAgentOptimizer, ((train_loader, val_loader), LeGRPruner(model, compression_ctrl, 'L1'), model, 0.8,  100), {})

            # logger.info('COEFF = {}, LOSS = {},  reward = {}, acc = {}'.format(coeff, LOSS, reward, acc))
            # logger.info('Actions = {}'.format(actions))

            # print_statistics(compression_ctrl.statistics())
    # is_inception = 'inception' in model_name
    # train(config, compression_ctrl, model, criterion, is_inception, lr_scheduler, model_name, optimizer,
    #       train_loader, train_sampler, val_loader, best_acc1)
    #
    # top1, top_5 = validate(val_loader, model, criterion, config)


main(sys.argv[1:])
