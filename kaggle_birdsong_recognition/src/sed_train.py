from argparse import ArgumentParser, Namespace
from engine.sed_engine import SedEngine
import importlib
import torch
import ignite.distributed as idist
torch.backends.cudnn.benchmark = True


def run(local_rank, config):
    config.train_bs = 3
    pe = SedEngine(local_rank, config)
    pe.train(config.run_params)


def main_parallel(hyperparams):
    with idist.Parallel(**hyperparams.dist_params) as parallel:
        parallel.run(run, hyperparams)


def main(hyperparams):
    run(0, hyperparams)


def run_main():
    parser = ArgumentParser(parents=[])
    
    parser.add_argument('--config', type=str)
    
    # params = parser.parse_args()
    params = Namespace(config='config_params.example_config') 
    print("------------------------", type(params))
    print(params)
    print("----------------", type(params.config))
    module = importlib.import_module(params.config, package=None)
    hyperparams = module.Parameters()
    
    main(hyperparams)

run_main()

