import argparse
import contextlib
import json
import os
import shutil
import sys
from dataclasses import dataclass

import hippynn
import matplotlib
import numpy as np
import torch
from hippynn import plotting
from hippynn.additional import MAEPhaseLoss, MSEPhaseLoss, NACRNode
from hippynn.experiment import setup_and_train
from hippynn.experiment.controllers import PatienceController, RaiseBatchSizeOnPlateau
from hippynn.graphs import inputs, loss, networks, physics, targets

# increase the recursion limit for now
# sys.setrecursionlimit(756)
# if `tkagg` is default, plotting speed will be horrible
matplotlib.use("Agg")
# default types for torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_default_dtype(torch.float32)


@dataclass(repr=True)
class ArgsList:
    """
    List of all CLI/Ray arguments. For code intellisense only. As all attributes of
    argparser are defined dynamically, there is no way for the editor to know what they
    are.

    For more details, check https://github.com/microsoft/pylance-release/issues/628
    """

    tag: str
    gpu: int
    noprogress: bool
    custom_kernel: bool
    handle_work_dir: bool
    work_dir: str
    rerun: bool
    n_states: int
    n_atoms: int
    training_targets: json.loads
    log_filename: str
    possible_species: json.loads
    n_interactions: int
    n_atom_layers: int
    n_features: int
    n_sensitivities: int
    lower_cutoff: float
    upper_cutoff: float
    cutoff_distance: float
    dataset_location: str
    dataset_name: str
    split_ratio: json.loads
    seed: int
    n_workers: int
    plot_frequency: int
    init_batch_size: int
    max_batch_size: int
    init_learning_rate: float
    raise_batch_patience: int
    termination_patience: int
    max_epochs: int
    stopping_key: str


def build_network(network_params: dict):
    # input layer of the network
    species = inputs.SpeciesNode(db_name="Z")
    positions = inputs.PositionsNode(db_name="R")
    network = networks.Hipnn(
        "hipnn_model", (species, positions), module_kwargs=network_params
    )
    return species, positions, network


def energy_target(n_states: int, network: networks.Hipnn):
    outputs = []
    energy_nodes = []
    for i in range(n_states):
        name = f"E{i+1}"
        energy = targets.HEnergyNode(f"HEnergy{i+1}", network)
        # actual training target should be the main output, i.e., mol_energy
        mol_energy = energy.mol_energy
        mol_energy.db_name = name
        energy_nodes.append(energy)
        outputs.append(mol_energy)
    # output dictionary
    d = {
        "mse_loss_func": loss.MSELoss,
        "mae_loss_func": loss.MAELoss,
        "norm": 1,
        "loss_weight": 0.1,
        "outputs": outputs,
        "energy_nodes": energy_nodes,
    }
    return d


def dipole_target(
    n_states: int, network: networks.Hipnn, positions: inputs.PositionsNode
):
    outputs = []
    charge_nodes = []
    for i in range(n_states):
        name = f"D{i+1}"
        # obtain dipole from charges * positions
        charge = targets.HChargeNode(f"HCharge{i+1}", network)
        dipole = physics.DipoleNode(name, (charge, positions), db_name=name)
        charge_nodes.append(charge)
        outputs.append(dipole)
    # output dictionary
    d = {
        "mse_loss_func": MSEPhaseLoss,
        "mae_loss_func": MAEPhaseLoss,
        "norm": np.sqrt(3),
        "loss_weight": 1.0,
        "outputs": outputs,
        "charge_nodes": charge_nodes,
    }
    return d


def nacr_target(
    training_targets: dict,
    n_states: int,
    n_atoms: int,
    network: networks.Hipnn,
    positions: inputs.PositionsNode,
):
    training_targets["nacr"] = {
        "mse_loss_func": MSEPhaseLoss,
        "mae_loss_func": MAEPhaseLoss,
        "norm": np.sqrt(n_atoms * 3),
        "loss_weight": 1.0,
    }
    outputs = []
    # build the charge_nodes if dipole is not a target
    if "dipole" not in training_targets:
        charge_nodes = []
        for i in range(n_states):
            charge_nodes.append(targets.HChargeNode(f"HCharge{i+1}", network))
        training_targets["nacr"] = {"charge_nodes": charge_nodes}
    # otherwise take it from dipole's dictionary
    else:
        charge_nodes = training_targets["dipole"]["charge_nodes"]
    # obtain the energy nodes from energy's dictionary
    energy_nodes = training_targets["energy"]["energy_nodes"]
    for i in range(n_states):
        # energy nodes and dipole nodes can directly be reused here
        q_i = charge_nodes[i]
        e_i = energy_nodes[i]
        for j in range(i + 1, n_states):
            q_j = charge_nodes[j]
            e_j = energy_nodes[j]
            name = f"ScaledNACR_{i+1}_{j+1}"
            nacr = NACRNode(name, (q_i, q_j, positions, e_i, e_j), db_name=name)
            outputs.append(nacr)
    training_targets["nacr"]["outputs"] = outputs
    return training_targets


def build_output_layer(
    params: ArgsList, network: networks.Hipnn, positions: inputs.PositionsNode
):
    n_states = params.n_states
    training_targets = {}
    train_nacr = False
    # sort the targets so energy or dipole should show up before nacr if exists
    for t in sorted(params.training_targets):
        # normalize all letters in the targets to lower case
        t = t.lower()
        # NACR need to be treated separately
        if t == "nacr":
            if n_states == 1:
                print("At least 2 states needed to train NACR.")
            elif "energy" in training_targets:
                train_nacr = True
            else:
                sys.exit("To train NACR, energies must be in the targets as well.")
        # add other targets to the dictionary directly
        else:
            if t == "energy":
                training_targets[t] = energy_target(n_states, network)
            elif t == "dipole":
                training_targets[t] = dipole_target(n_states, network, positions)
            else:
                print(f"Unknown target {t}")

    if len(training_targets) == 0:
        sys.exit("No suitable targets")
    # NACR is treated separately
    if train_nacr:
        training_targets = nacr_target(
            training_targets, n_states, params.n_atoms, network, positions
        )
    return training_targets


def build_loss(training_targets: dict, network: networks.Hipnn):
    # TODO: added weights for different states and targets
    validation_losses = {}

    for i, _ in enumerate(training_targets.items()):
        k, v = _
        outputs = v["outputs"]
        norm = v["norm"]
        weight = v["loss_weight"]
        mse_loss_func = v["mse_loss_func"]
        mae_loss_func = v["mae_loss_func"]
        # per node RMSE and MAE
        # also accumulate the total RMSE and MAE for this target
        for j, node in enumerate(outputs):
            rmse = mse_loss_func.of_node(node) ** 0.5
            validation_losses[f"{node.db_name}-RMSE"] = rmse
            mae = mae_loss_func.of_node(node)
            validation_losses[f"{node.db_name}-MAE"] = mae
            if j == 0:
                target_rmse = rmse
                target_mae = mae
            else:
                target_rmse += rmse
                target_mae += mae
        validation_losses[f"{k.upper()}-RMSE"] = target_rmse
        validation_losses[f"{k.upper()}-MAE"] = target_mae
        if norm != 1.0:
            target_rmse /= norm
        target_loss = target_rmse + target_mae
        validation_losses[f"{k.upper()}-Loss"] = target_loss
        if weight != 1.0:
            target_loss *= weight
        if i == 0:
            total_loss = target_loss
        else:
            total_loss += target_loss
    # l2 regularization
    l2_reg = loss.l2reg(network)
    # TODO: this pre-factor should be a variable
    loss_regularization = 2e-6 * l2_reg
    # add total loss to the dictionary
    validation_losses["Loss"] = total_loss + loss_regularization
    training_targets["Losses"] = validation_losses
    return training_targets


def setup_plots(
    training_targets: dict, network: networks.Hipnn, n_interactions: int, freq: int
):

    node_plots = []
    for _, v in training_targets.items():
        outputs = v["outputs"]
        for node in outputs:
            node_plots.append(plotting.Hist2D.compare(node, saved=True, shown=False))
    for i in range(n_interactions):
        node_plots.append(
            plotting.SensitivityPlot(
                network.torch_module.sensitivity_layers[i],
                saved=f"Sensitivity_{i}.pdf",
                shown=False,
            )
        )

    return plotting.PlotMaker(*node_plots, plot_every=freq)


def setup_experiment(losses: dict, plotter: plotting.PlotMaker, params: ArgsList):
    # Assemble Pytorch Model that's actually trained.
    training_modules, db_info = hippynn.experiment.assemble_for_training(
        losses["Loss"],
        losses,
        plot_maker=plotter,
    )
    # Parameters describing the training procedure.
    optimizer = torch.optim.Adam(
        training_modules.model.parameters(), lr=params.init_learning_rate
    )
    batch_size = params.max_batch_size
    scheduler = RaiseBatchSizeOnPlateau(
        optimizer=optimizer,
        max_batch_size=batch_size,
        patience=params.raise_batch_patience,
        factor=0.5,
    )

    controller = PatienceController(
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=params.init_batch_size,
        eval_batch_size=batch_size,
        max_epochs=params.max_epochs,
        stopping_key=params.stopping_key,
        fraction_train_eval=0.1,
        termination_patience=params.termination_patience,
    )

    experiment_params = hippynn.experiment.SetupParams(
        controller=controller,
        device=params.gpu,
    )
    return training_modules, db_info, experiment_params


def load_database(params: ArgsList, db_info: dict):
    database = hippynn.databases.DirectoryDatabase(
        name=params.dataset_name,  # Prefix for arrays in the directory
        directory=params.dataset_location,
        test_size=params.split_ratio[0],  # Fraction or number of samples to test on.
        valid_size=params.split_ratio[
            1
        ],  # Fraction or number of samples to validate on
        seed=params.seed,  # Random seed for splitting data
        **db_info,  # Adds the inputs and targets db_names from the model as things to load
    )
    database.send_to_device(params.gpu)
    return database


def main(params: ArgsList):
    """_summary_

    Args:
        params (ArgsList): input parameters. Actually an `argparse.Namespace`. The type\
            `ArgsList` is used to fool the editor for auto-completion.

    Returns:
        dict: key parameters and metric of the model
    """
    # global hippynn settings
    if params.noprogress:
        hippynn.settings.PROGRESS = None
    hippynn.custom_kernels.set_custom_kernels(params.custom_kernel)
    hippynn.settings.WARN_LOW_DISTANCES = False

    # Hyperparameters for the network
    n_interactions = params.n_interactions
    network_params = {
        "possible_species": params.possible_species,
        "n_features": params.n_features,
        "n_sensitivities": params.n_sensitivities,
        "dist_soft_min": params.lower_cutoff,
        "dist_soft_max": params.upper_cutoff,
        "dist_hard_max": params.cutoff_distance,
        "n_interaction_layers": n_interactions,
        "n_atom_layers": params.n_atom_layers,
    }
    # dump parameters to the log file
    print(json.dumps(network_params, indent=4))

    _, positions, network = build_network(network_params)
    training_targets = build_output_layer(params, network, positions)
    freq = params.plot_frequency
    if freq > 0:
        plotter = setup_plots(training_targets, network, n_interactions, freq)
    else:
        plotter = None
    training_targets = build_loss(training_targets, network)
    print(training_targets["Losses"].keys())
    training_modules, db_info, experiment_params = setup_experiment(
        training_targets["Losses"], plotter, params
    )
    database = load_database(params, db_info)

    metric_tracker = setup_and_train(
        training_modules=training_modules,
        database=database,
        setup_params=experiment_params,
    )

    del network_params["possible_species"]
    network_params["metric"] = metric_tracker.best_metric_values
    network_params["avg_epoch_time"] = np.average(metric_tracker.epoch_times)
    network_params["Loss"] = metric_tracker.best_metric_values["valid"]["Loss"]

    with open("training_summary.json", "w") as out:
        json.dump(network_params, out, indent=4)
    return network_params


def path_handler(
    params: ArgsList,
    naming=[
        "n_features",
        "n_sensitivities",
        "lower_cutoff",
        "upper_cutoff",
        "cutoff_distance",
        "n_interactions",
        "n_atom_layers",
    ],
):
    """
    Handling the path for current experiment. Only used when `handle_work_dir=True`. \
    If the script is called from `ray`, handling paths yourself will be problematic.

    Args:
        params (ArgsList): list of parameters
        naming (list, optional): parameters used to construct the directory's name \
            concatenated by underscores. Defaults to ["n_features", "n_sensitivities",\
            "lower_cutoff", "upper_cutoff", "cutoff_distance", "n_interactions", \
            "n_atom_layers"].

    Raises:
        Exception: anything that indicates the model should be rerun

    Returns:
        dict: finished results
    """
    dir_name = params.tag
    for i in naming:
        dir_name += f"_{getattr(params, i)}"
    if not os.path.exists(params.work_dir):
        os.mkdir(params.work_dir)
    os.chdir(params.work_dir)
    # keep a note on the name of current running model
    with open("folder", "w") as f:
        print(dir_name, file=f)
    # create the folder if it doesn't exist
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    # if folder already exist and retraining is not enforced
    elif not params.rerun:
        try:
            with open(dir_name + "/training_summary.json", "r") as out:
                tmp = json.load(out)
                if len(tmp) >= 7:
                    print(f"{dir_name} already finished")
                    return tmp
                else:
                    raise Exception("training_summary.json is incomplete.")
        # if any exception is raised, cleanup everything in the folder
        except Exception as e:
            shutil.rmtree(dir_name)
            os.mkdir(dir_name)
    # if retrain is enforced
    else:
        shutil.rmtree(dir_name)
        os.mkdir(dir_name)
    os.chdir(dir_name)


class CustomFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter
):
    """Formatter for the help message of the argument parser"""

    pass


def read_args(
    tag="test",
    gpu=0,
    noprogress=False,
    custom_kernel=False,
    handle_work_dir=False,
    work_dir="test",
    rerun=True,
    n_states=5,
    n_atoms=6,
    training_targets=["energy", "dipole", "nacr"],
    log_filename="training_log.txt",
    possible_species=[0, 1, 6],
    n_interactions=3,
    n_atom_layers=3,
    n_features=15,
    n_sensitivities=20,
    lower_cutoff=0.8,
    upper_cutoff=20.0,
    cutoff_distance=24.0,
    dataset_location="/projects/ml4chem/xinyang/ethene_with_nacr/dataset",
    dataset_name="eth_",
    split_ratio=[0.3, 0.2],
    seed=7777,
    n_workers=1,
    plot_frequency=100,
    init_batch_size=32,
    max_batch_size=2048,
    init_learning_rate=1e-3,
    raise_batch_patience=96,
    termination_patience=500,
    max_epochs=3000,
    stopping_key="Loss",
    bypass_cli_args=False,
):
    """
    Function to read CLI arguments. The keyword arguments are used to pass the default
    values to the function. Note that to comply with POSIX standard, the long arguments
    have hyphens, instead of underscores.

    Args:
        bypass_cli_args (bool, optional): directly assume a namespace with passed in \
            arguments if true.

    Returns:
        ArgsList: parsed arguments.
    """

    if bypass_cli_args:
        args = vars()
        del args["bypass_cli_args"]
        return ArgsList(**args)

    parser = argparse.ArgumentParser(formatter_class=CustomFormatter)

    # global settings of the task
    parser.add_argument("--tag", type=str, default=tag, help="name for run")
    parser.add_argument("--gpu", type=int, default=gpu, help="which GPU to run on")
    parser.add_argument(
        "--noprogress",
        action="store_true",
        default=noprogress,
        help="suppress the progress bars if the argument exists",
    )
    parser.add_argument(
        "--custom-kernel",
        action="store_true",
        default=custom_kernel,
        help="enable custom kernels if the argument exists",
    )
    parser.add_argument(
        "--handle-work-dir",
        action="store_true",
        default=handle_work_dir,
        help="the script will handle the working directory if the argument exists",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=work_dir,
        help=(
            "root directory for all tests\n"
            "each test will have its own subfolder\n"
            "only works when handle_work_dir=True"
        ),
    )
    parser.add_argument(
        "--rerun",
        action="store_false",
        default=rerun,
        help="retrain the model if the argument and path to the model exist",
    )
    parser.add_argument(
        "--n-states",
        type=int,
        default=n_states,
        help="number of excited states included in the training",
    )
    parser.add_argument(
        "--n-atoms",
        type=int,
        default=n_atoms,
        help="number of atoms in the molecule",
    )
    parser.add_argument(
        "--training-targets",
        type=json.loads,
        default=training_targets,
        help="a quoted list with the target quantities to train",
    )
    parser.add_argument(
        "--log-filename",
        type=str,
        default=log_filename,
        help="filename to save the training log",
    )

    # network parameters
    parser.add_argument(
        "--possible-species",
        type=json.loads,
        default=possible_species,
        help=(
            "a quoted list of possible species in the dataset\n"
            "a padding '0' should always be in the list"
        ),
    )
    parser.add_argument(
        "--n-interactions",
        type=int,
        default=n_interactions,
        help="number of interaction layers",
    )
    parser.add_argument(
        "--n-atom-layers", type=int, default=n_atom_layers, help="number of atom layers"
    )
    parser.add_argument(
        "--n-features", type=int, default=n_features, help="number of neurons per layer"
    )
    parser.add_argument(
        "--n-sensitivities",
        type=int,
        default=n_sensitivities,
        help="number of radial distribution functions",
    )

    # distances for radial functions
    parser.add_argument(
        "--lower-cutoff",
        type=float,
        default=lower_cutoff,
        help="where to initialize the shortest distance sensitivity",
    )
    parser.add_argument(
        "--upper-cutoff",
        type=float,
        default=upper_cutoff,
        help="where to initialize the longest distance sensitivity",
    )
    parser.add_argument(
        "--cutoff-distance",
        type=float,
        default=cutoff_distance,
        help="cutoff distance where all sensitivities go to 0",
    )

    # dataset parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=seed,
        help="random seed for initialization and dataset split",
    )
    parser.add_argument(
        "--dataset-location",
        type=str,
        default=dataset_location,
        help="path to the folder contains the dataset",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=dataset_name,
        help="prefix for the .npy file in the dataset",
    )
    parser.add_argument(
        "--split-ratio",
        type=json.loads,
        default=split_ratio,
        help=(
            "a quoted list of split ratio for the test and validation set\n"
            "can be number of points (integers) or ratios (fraction numbers)"
        ),
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=n_workers,
        help="workers for pytorch dataloader",
    )

    # training parameters
    parser.add_argument(
        "--plot-frequency",
        type=int,
        default=plot_frequency,
        help=(
            "frequency (number of epochs) to plot the histograms\n0 to disable plotting"
        ),
    )
    parser.add_argument(
        "--init-batch-size",
        type=int,
        default=init_batch_size,
        help="initial batch size",
    )
    parser.add_argument(
        "--max-batch-size", type=int, default=max_batch_size, help="maximum batch size"
    )
    parser.add_argument(
        "--init-learning-rate",
        type=float,
        default=init_learning_rate,
        help="initial learning rate",
    )
    parser.add_argument(
        "--raise-batch-patience",
        type=int,
        default=raise_batch_patience,
        help="maximum plateau epoches before raising the batch size",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=max_epochs,
        help="maximum number of total epochs",
    )
    parser.add_argument(
        "--stopping-key",
        type=str,
        default=stopping_key,
        help="criteria for early stopping of training",
    )
    parser.add_argument(
        "--termination-patience",
        type=str,
        default=termination_patience,
        help="number of plateau epochs before early stopping the training process",
    )

    args = parser.parse_args(bypass_cli_args, namespace=ArgsList)
    return args


if __name__ == "__main__":
    params = read_args(
        handle_work_dir=True,
        n_states=5,
        upper_cutoff=10,
        init_batch_size=512,
        split_ratio=[0.3, 0.2],
    )
    if params.handle_work_dir:
        results = path_handler(params)
        if results:
            print(json.dumps(results, indent=4))
            sys.exit(0)
    with open(params.log_filename, "w") as log_file:
        # this way is preferred than `hippynn.tools.log_terminal`
        # because the terminal output will be completely suppressed now
        # which could result in a huge SLURM log file if many searches are performed
        with contextlib.redirect_stdout(log_file):
            main(params)
