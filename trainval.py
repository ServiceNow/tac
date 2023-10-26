import os
import argparse
import tqdm
import torch
import exp_configs

from src import datasets, models
from src.constants import DEVICE
from haven import haven_wizard as hw

RESULTS_FNAME = "results.ipynb"


def trainval(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """

    torch.backends.cudnn.benchmark = True

    # Create data loader and model
    train_set = datasets.get_dataset(
        name=exp_dict["dataset"],
        split="train",
        datadir=args.data_path,
    )
    test_set = datasets.get_dataset(
        name=exp_dict["dataset"],
        split="test",
        datadir=args.data_path,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=exp_dict["train_batch_size"],
        num_workers=exp_dict["n_workers"],
        shuffle=True,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=exp_dict["test_batch_size"],
        num_workers=exp_dict["n_workers"],
        shuffle=True,
        pin_memory=True,
    )

    # Configures scheduler with the correct number of iterations rather than epochs.
    if "max_epochs" in exp_dict["scheduler_config"]["kwargs"]:
        exp_dict["scheduler_config"]["kwargs"]["max_epochs"] = args.epochs * len(
            train_loader
        )
    if "warmup_epochs" in exp_dict["scheduler_config"]["kwargs"]:
        exp_dict["scheduler_config"]["kwargs"]["warmup_epochs"] = exp_dict[
            "scheduler_config"
        ]["kwargs"]["warmup_epochs"] * len(train_loader)

    model = models.get_model(exp_dict=exp_dict)

    # Resume or initialize checkpoint
    cm = hw.CheckpointManager(savedir)
    state_dict = cm.load_model()
    if state_dict is not None:
        model.set_state_dict(state_dict)
    elif exp_dict["pretrained_path"] is not None:
        pretrained_state_dict = torch.load(exp_dict["pretrained_path"])
        model.base_model.load_state_dict(pretrained_state_dict["model"], strict=False)
        model.classification_head.load_state_dict(
            pretrained_state_dict["classification_head"]
        )
        model = model.to(DEVICE)

    model = model.to(DEVICE)

    best_model_metric_name = "error_detection_auc"

    # Train and Validate
    for epoch in tqdm.tqdm(
        range(cm.get_epoch(), args.epochs), desc="Running Experiment"
    ):

        # Train for one epoch
        train_dict = model.train_on_loader(train_loader, epoch)
        test_dict = model.eval_on_loader(test_loader)

        # Get Metrics
        score_dict = {
            "epoch": epoch,
        }

        score_dict.update(train_dict)
        score_dict.update(test_dict)

        # Save Metrics in "savedir" as score_list.pkl
        cm.log_metrics(score_dict)
        states = model.get_state_dict()
        torch.save(states, os.path.join(savedir, "model.pth"))

        if score_dict[best_model_metric_name] > model.best_val_score:
            model.best_val_score = score_dict[best_model_metric_name]
            torch.save(states, os.path.join(savedir, "model_best_acc.pth"))
    print("Experiment done\n")


if __name__ == "__main__":
    # Specify arguments regarding save directory and job scheduler
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--exp_group",
        help="Define the experiment group to run. Adding the string selfsupervised_ to this arg will run the self supervised version of the experiment.",
    )

    parser.add_argument(
        "-sb",
        "--savedir_base",
        required=True,
        help="Define the base directory where the experiments will be saved.",
    )
    parser.add_argument(
        "-d",
        "--data_path",
        default="base",
        help="Define the base directory where data will be cached.",
    )
    parser.add_argument(
        "-r", "--reset", default=0, type=int, help="Reset or resume the experiment."
    )
    parser.add_argument(
        "-j", "--job_scheduler", default=None, help="Choose Job Scheduler."
    )
    parser.add_argument(
        "--python_binary", default="python", help="path to your python executable"
    )
    parser.add_argument(
        "--epochs", default=10, type=int, help="Number of epochs to train."
    )

    args, others = parser.parse_known_args()

    if args.data_path == "base":
        args.data_path = os.path.dirname(args.savedir_base)

    print(args)

    # Choose Job Scheduler
    job_config = None

    # Run experiments and create results file
    hw.run_wizard(
        func=trainval,
        exp_list=exp_configs.EXP_GROUPS[args.exp_group],
        savedir_base=args.savedir_base,
        reset=args.reset,
        job_config=job_config,
        results_fname="results.ipynb",
        python_binary_path=args.python_binary,
        args=args,
    )
