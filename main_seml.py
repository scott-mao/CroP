import sys
import warnings

from models import GeneralModel
from models.statistics.Metrics import Metrics
from utils.config_utils import *
from utils.model_utils import *
from utils.system_utils import *
# from rigl_torch.RigL import RigLScheduler

warnings.filterwarnings("ignore")

import logging
from sacred import Experiment
import numpy as np
import seml


ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

def main(
        arguments,
        metrics: Metrics
):
    if arguments.disable_autoconfig:
        autoconfig(arguments)

    global out
    out = metrics.log_line
    out(f"starting at {get_date_stamp()}")

    # hardware
    device = configure_device(arguments)

    if arguments.disable_cuda_benchmark:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # for reproducibility
    configure_seeds(arguments, device)

    # filter for incompatible properties
    assert_compatibilities(arguments)
    # get model
    model: GeneralModel = find_right_model(
        NETWORKS_DIR, arguments.model,
        device=device,
        hidden_dim=arguments.hidden_dim,
        input_dim=arguments.input_dim,
        output_dim=arguments.output_dim,
        is_maskable=arguments.disable_masking,
        is_tracking_weights=arguments.track_weights,
        is_rewindable=arguments.enable_rewinding,
        is_growable=arguments.growing_rate > 0,
        outer_layer_pruning=arguments.outer_layer_pruning,
        maintain_outer_mask_anyway=(
                                       not arguments.outer_layer_pruning) and (
                                           "Structured" in arguments.prune_criterion),
        l0=arguments.l0,
        l0_reg=arguments.l0_reg,
        N=arguments.N,
        beta_ema=arguments.beta_ema,
        l2_reg=arguments.l2_reg
    ).to(device)

    # get criterion
    criterion = find_right_model(
        CRITERION_DIR, arguments.prune_criterion,
        model=model,
        limit=arguments.pruning_limit,
        start=0.5,
        steps=arguments.snip_steps,
        device=arguments.pruning_device
    )

    # load pre-trained weights if specified
    load_checkpoint(arguments, metrics, model)

    # load data
    train_loader, test_loader = find_right_model(
        DATASETS, arguments.data_set,
        arguments=arguments
    )

    # get loss function
    loss = find_right_model(
        LOSS_DIR, arguments.loss,
        device=device,
        l1_reg=arguments.l1_reg,
        lp_reg=arguments.lp_reg,
        l0_reg=arguments.l0_reg,
        hoyer_reg=arguments.hoyer_reg
    )

    # get optimizer
    optimizer = find_right_model(
        OPTIMS, arguments.optimizer,
        params=model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.l2_reg if not arguments.l0 else 0,
        # momentum=arguments.momentum if arguments.momentum else 0
    )
    from torch.optim.lr_scheduler import StepLR, OneCycleLR
    if arguments.model == 'VGG16' or arguments.prune_criterion != 'EmptyCrit':
        scheduler = StepLR(optimizer, step_size=30000, gamma=0.2)
    elif arguments.prune_criterion == 'EmptyCrit':
        scheduler = OneCycleLR(optimizer, max_lr=arguments.learning_rate,
                                     steps_per_epoch=len(train_loader), epochs=arguments.epochs)
    # now, create the RigLScheduler object
    # pruner = RigLScheduler(model,
    #                        optimizer,
    #                        dense_allocation=0.1,
    #                        sparsity_distribution='uniform',
    #                        T_end=5859,
    #                        delta=100,
    #                        alpha=0.3,
    #                        grad_accumulation_n=1,
    #                        static_topo=False,
    #                        ignore_linear_layers=False,
    #                        state_dict=None)
    run_name = f'_model={arguments.model}_dataset={arguments.data_set}_prune-criterion={arguments.prune_criterion}' + \
               f'_pruning-limit={arguments.pruning_limit}_train-scheme={arguments.train_scheme}_seed={arguments.seed}'
    if not arguments.eval:

        # build trainer
        trainer = find_right_model(
            TRAINERS_DIR, arguments.train_scheme,
            model=model,
            loss=loss,
            optimizer=optimizer,
            device=device,
            arguments=arguments,
            train_loader=train_loader,
            test_loader=test_loader,
            metrics=metrics,
            criterion=criterion,
            scheduler=scheduler,
            run_name=run_name
            # pruner=pruner
        )

        from codecarbon import EmissionsTracker
        tracker = EmissionsTracker()
        tracker.start()
        trainer.train()
        emissions: float = tracker.stop()

    else:

        tester = find_right_model(
            TESTERS_DIR, arguments.test_scheme,
            train_loader=train_loader,
            test_loader=test_loader,
            model=model,
            loss=loss,
            optimizer=optimizer,
            device=device,
            arguments=arguments,
        )

        return tester.evaluate()

    out(f"finishing at {get_date_stamp()}")
    results = {}
    results['emissions'] = emissions
    results['run_name'] = run_name
    results['crit'] = np.array([i.detach().cpu() for i in trainer.crit])
    results['test_acc'] = trainer.test_acc
    results['train_acc'] = trainer.train_acc
    results['train_loss'] = trainer.train_loss
    results['test_loss'] = trainer.test_loss
    results['sparse_weight'] = trainer.sparse_weight
    results['sparse_node'] = trainer.sparse_node
    results['sparse_hm'] = trainer.sparse_hm
    results['sparse_log_disk_size'] = trainer.sparse_log_disk_size
    results['flops_per_sample'] = trainer.flops_per_sample
    results['flops_log_cum'] = trainer.flops_log_cum
    results['gpu_ram'] = trainer.gpu_ram
    results['max_gpu_ram'] = trainer.max_gpu_ram
    results['batch_time'] = trainer.batch_time
    results['gpu_time'] = trainer.time_gpu
    return results

def assert_compatibilities(arguments):
    check_incompatible_props([arguments.loss != "L0CrossEntropy", arguments.l0], "l0", arguments.loss)
    check_incompatible_props([arguments.train_scheme != "L0Trainer", arguments.l0], "l0", arguments.train_scheme)
    check_incompatible_props([arguments.l0, arguments.group_hoyer_square, arguments.hoyer_square],
                             "Choose one mode, not multiple")
    # check_incompatible_props(
    #     ["Structured" in arguments.prune_criterion, "Group" in arguments.prune_criterion, "ResNet" in arguments.model],
    #     "structured", "residual connections")
    # todo: add more


def load_checkpoint(arguments, metrics, model):
    if (not (arguments.checkpoint_name is None)) and (not (arguments.checkpoint_model is None)):
        path = os.path.join(RESULTS_DIR, arguments.checkpoint_name, MODELS_DIR, arguments.checkpoint_model)
        state = DATA_MANAGER.load_python_obj(path)
        try:
            model.load_state_dict(state)
        except KeyError as e:
            print(list(state.keys()))
            raise e
        out(f"Loaded checkpoint {arguments.checkpoint_name} from {arguments.checkpoint_model}")


def log_start_run(arguments, metrics):
    arguments.PyTorch_version = torch.__version__
    arguments.PyThon_version = sys.version
    arguments.pwd = os.getcwd()
    metrics.log_line("PyTorch version:", torch.__version__, "Python version:", sys.version)
    metrics.log_line("Working directory: ", os.getcwd())
    metrics.log_line("CUDA avalability:", torch.cuda.is_available(), "CUDA version:", torch.version.cuda)
    metrics.log_line(arguments)


def get_arguments():
    global arguments
    arguments = parse()
    if arguments.disable_autoconfig:
        autoconfig(arguments)
    return arguments

from dataclasses import dataclass

@ex.automain
def run(arguments):
    @dataclass
    class args:
        eval_freq = arguments['eval_freq']  # evaluate every n batches
        save_freq = arguments['save_freq']  # save model every n epochs, besides before and after training
        batch_size = arguments['batch_size']  # 128  # size of batches, for Imagenette 128
        seed =  arguments['seed']  # random seed
        max_training_minutes = arguments['max_training_minutes']  # one hour and a 45 minutes max, process killed after n minutes (after finish of epoch)
        plot_weights_freq = arguments['plot_weights_freq']  # plot pictures to tensorboard every n epochs
        prune_freq = arguments['prune_freq']  # if pruning during training: how long to wait between pruning events after first pruning
        prune_delay = arguments['prune_delay']  # "if pruning during training: how long to wait before pruning first time
        lower_limit = arguments['lower_limit']
        epochs = arguments['epochs']
        rewind_to = arguments['rewind_to']  # rewind to this epoch if rewinding is done
        hidden_dim = arguments['hidden_dim']
        input_dim = arguments['input_dim']
        output_dim = arguments['output_dim']
        N = arguments['N']  # size of dataset (used for l0)
        snip_steps = arguments['snip_steps']  # 's' in algorithm box, number of pruning steps for 'rule of thumb', TODO
        pruning_rate = arguments['pruning_rate']  # pruning rate passed to criterion at pruning event. however, most override this
        growing_rate = arguments['growing_rate']  # grow back so much every epoch (for future criterions)
        pruning_limit = arguments['pruning_limit'] # Prune until here, if structured in nodes, if unstructured in weights. most criterions use this instead of the pruning_rate
        prune_to = arguments['prune_to']
        learning_rate = arguments['learning_rate']
        grad_clip = arguments['grad_clip']
        grad_noise = arguments['grad_noise']  # added gaussian noise to gradients
        l2_reg = arguments['l2_reg']  # weight decay
        l1_reg = arguments['l1_reg']  # l1-norm regularisation
        lp_reg = arguments['lp_reg']  # lp regularisation with p < 1
        l0_reg = arguments['l0_reg']  # l0 reg lambda hyperparam
        hoyer_reg = arguments['hoyer_reg']  # hoyer reg lambda hyperparam
        beta_ema = arguments['beta_ema']  # l0 reg beta ema hyperparam

        loss = arguments['loss']
        optimizer = arguments['optimizer']
        model = arguments['model']  # WideResNet28x10  # ResNet not supported with structured
        data_set = arguments['data_set']
        prune_criterion = arguments['prune_criterion']  # HoyerSquare is one shot, pruning limit doesn't matter in this implementation
        train_scheme = arguments['train_scheme']  # default: DefaultTrainer

        device = arguments['device']
        structured_prior = arguments['structured_prior']
        pruning_device = arguments['pruning_device']

        checkpoint_name = arguments['checkpoint_name']
        checkpoint_model = arguments['checkpoint_model']

        disable_cuda_benchmark = arguments['disable_cuda_benchmark']  # speedup (disable) vs reproducibility (leave it)
        eval = arguments['eval']
        disable_autoconfig = arguments['disable_autoconfig']  # for the brave
        preload_all_data = arguments['preload_all_data']  # load all data into ram memory for speedups
        tuning = arguments['tuning']  # splits trainset into train and validationset, omits test set


        track_weights = arguments['track_weights']  # "keep statistics on the weights through training
        disable_masking = arguments['disable_masking']  # disable the ability to prune unstructured
        enable_rewinding = arguments['enable_rewinding']  # enable the ability to rewind to previous weights
        outer_layer_pruning = arguments['outer_layer_pruning']  # allow to prune outer layers (unstructured) or not (structured)
        random_shuffle_labels = arguments['random_shuffle_labels']  # run with random-label experiment from zhang et al
        l0 = arguments['l0']  # run with l0 criterion, might overwrite some other arguments
        hoyer_square = arguments['hoyer_square']  # "run in unstructured DeephoyerSquare criterion, might overwrite some other arguments
        group_hoyer_square = arguments['group_hoyer_square']  # run in unstructured Group-DeephoyerSquare criterion, might overwrite some other arguments

        disable_histograms = arguments['disable_histograms']
        disable_saliency = arguments['disable_saliency']
        disable_confusion = arguments['disable_confusion']
        disable_weightplot = arguments['disable_weightplot']
        disable_netplot = arguments['disable_netplot']
        skip_first_plot = arguments['skip_first_plot']

    metrics = Metrics()
    out = metrics.log_line
    print = out
    ensure_current_directory()
    # get_arguments()
    log_start_run(args, metrics)
    out("\n\n")
    metrics._batch_size = args.batch_size
    metrics._eval_freq = args.eval_freq

    results = main(args, metrics)

    # the returned result will be written into the database
    return results
