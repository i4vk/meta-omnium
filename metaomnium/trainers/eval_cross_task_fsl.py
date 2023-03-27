# Our code builds on https://github.com/ihsaan-ullah/meta-album

# -----------------------
# Imports
# -----------------------

import os
import sys

# In order to import modules from packages in the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import argparse
import datetime
import json

import pickle
import random
import time
from copy import deepcopy

import numpy as np
import scipy.stats as st
import torch
import torch.nn as nn
from metaomnium.dataloaders.data_loader_cross_problem_fsl import (
    DataLoaderCrossProblem,
    create_datasets,
    create_datasets_task_type,
    process_labels,
    get_k_keypoints,
)
from metaomnium.models.finetuning import FineTuning
from metaomnium.models.proto_finetuning import ProtoFineTuning
from metaomnium.models.fsl_resnet import ResNet
from metaomnium.models.maml import MAML
from metaomnium.models.proto_maml import ProtoMAML
from metaomnium.models.meta_curvature import MetaCurvature
from metaomnium.models.protonet import PrototypicalNetwork
from metaomnium.models.ddrr import DDRR
from metaomnium.models.train_from_scratch import TrainFromScratch

from utils.configs import (
    FT_CONF,
    PROTO_FT_CONF,
    MAML_CONF,
    PROTO_MAML_CONF,
    METACURVATURE_CONF,
    MATCHING_CONF,
    PROTO_CONF,
    DDRR_CONF,
    TFS_CONF,
)
from utils.utils import *


# -----------------------
# Arguments
# -----------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument(
        "--train_datasets",
        type=str,
        required=True,
        help="datasets that were used for training. "
        + "Multiple datasets must be separeted by comma, e.g., BRD,CRS,FLW.",
    )
    parser.add_argument(
        "--val_id_datasets",
        type=str,
        required=False,
        default="",
        help="datasets that will be used for in-domain validation. "
        + "Multiple datasets must be separeted by comma, e.g., BRD,CRS,FLW.",
    )
    parser.add_argument(
        "--val_od_datasets",
        type=str,
        required=False,
        default="",
        help="datasets that will be used for out-domain validation. "
        + "Multiple datasets must be separeted by comma, e.g., BRD,CRS,FLW.",
    )
    parser.add_argument(
        "--test_id_datasets",
        type=str,
        required=False,
        default="",
        help="datasets that will be used for in-domain testing. "
        + "Multiple datasets must be separeted by comma, e.g., BRD,CRS,FLW",
    )
    parser.add_argument(
        "--test_od_datasets",
        type=str,
        required=False,
        default="",
        help="datasets that will be used for out-domain testing. "
        + "Multiple datasets must be separeted by comma, e.g., BRD,CRS,FLW",
    )
    parser.add_argument(
        "--model",
        choices=[
            "tfs",
            "finetuning",
            "proto_finetuning",
            "maml",
            "protomaml",
            "metacurvature",
            "protonet",
            "matchingnet",
            "ddrr",
        ],
        required=True,
        help="which model to use",
    )

    # Optional arguments

    # Train/valid episodes config
    parser.add_argument(
        "--n_way_train",
        type=int,
        default=5,
        help="number of ways for the support set of the training episodes, "
        + "if None, the episodes are any-way tasks. Default: 5.",
    )
    parser.add_argument(
        "--k_shot_train",
        type=int,
        default=None,
        help="number of shots for the support set of the training episodes, "
        + "if None, the episodes are any-shot tasks. Default: None.",
    )
    parser.add_argument(
        "--min_shots_train",
        type=int,
        default=1,
        help="minimum number of shots for the any-shot training tasks. "
        + "Default: 1.",
    )
    parser.add_argument(
        "--max_shots_train",
        type=int,
        default=5,
        help="minimum number of shots for the any-shot training tasks. "
        + "Default: 5.",
    )
    parser.add_argument(
        "--query_size_train",
        type=int,
        default=5,
        help="number of images for the query set of the training episodes. "
        + "Default: 5.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=None,
        help="size of minibatches for training only applies for flat batch "
        + "models. Default: None.",
    )

    # Test episodes config
    parser.add_argument(
        "--n_way_eval",
        type=int,
        default=None,
        help="number of ways for the support set of the testing episodes, "
        + "if None, the episodes are any-way tasks. Default: None.",
    )
    parser.add_argument(
        "--k_shot_eval",
        type=int,
        default=None,
        help="number of shots for the support set of the testing episodes, "
        + "if None, the episodes are any-shot tasks. Default: None.",
    )
    parser.add_argument(
        "--min_ways_eval",
        type=int,
        default=2,
        help="minimum number of ways for the any-way testing tasks. " + "Default: 2.",
    )
    parser.add_argument(
        "--max_ways_eval",
        type=int,
        default=20,
        help="maximum number of ways for the any-way testing tasks. " + "Default: 20.",
    )
    parser.add_argument(
        "--min_shots_eval",
        type=int,
        default=1,
        help="minimum number of shots for the any-shot testing tasks. " + "Default: 1.",
    )
    parser.add_argument(
        "--max_shots_eval",
        type=int,
        default=20,
        help="minimum number of shots for the any-shot testing tasks. "
        + "Default: 20.",
    )
    parser.add_argument(
        "--query_size_eval",
        type=int,
        default=5,
        help="number of images for the query set of the testing episodes. "
        + "Default: 5.",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=None,
        help="size of minibatches for testing only applies for flat-batch "
        + "models. Default: None.",
    )

    # Model configs
    parser.add_argument(
        "--runs", type=int, default=1, help="number of runs to perform. Default: 1."
    )
    parser.add_argument(
        "--train_iters",
        type=int,
        default=None,
        help="number of meta-training iterations. Default: None.",
    )
    parser.add_argument(
        "--eval_iters",
        type=int,
        default=600,
        help="number of meta-valid/test iterations. Default: 600.",
    )
    parser.add_argument(
        "--val_after",
        type=int,
        default=2500,
        help="after how many episodes the meta-validation should be "
        + "performed. Default: 2500.",
    )
    parser.add_argument(
        "--seed", type=int, default=1337, help="random seed to use. Default: 1337."
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="validate performance on meta-validation tasks. Default: True.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        help="backbone to use. Default: resnet18.",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        default=False,
        help="whether to freeze the weights in the finetuning model of "
        + "earlier layers. Default: False.",
    )
    parser.add_argument(
        "--meta_batch_size",
        type=int,
        default=1,
        help="number of tasks to compute outer-update. Default: 1.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="learning rate for (meta-)optimizer. Default: None.",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=None,
        help="number of weight updates per training set. Default: None.",
    )
    parser.add_argument(
        "--T_val",
        type=int,
        default=None,
        help="number of weight updates at validation time. Default: None.",
    )
    parser.add_argument(
        "--T_test",
        type=int,
        default=None,
        help="number of weight updates at test time. Default: None.",
    )
    parser.add_argument(
        "--base_lr",
        type=float,
        default=None,
        help="inner level learning rate: Default: None.",
    )
    parser.add_argument(
        "--test_lr",
        type=float,
        default=0.001,
        help="learning rate to use at meta-val/test time for finetuning. "
        + "Default: 0.001.",
    )
    parser.add_argument(
        "--test_opt",
        choices=["adam", "sgd"],
        default="adam",
        help="optimizer to use at meta-val/test time for finetuning. "
        + "Default: adam.",
    )
    parser.add_argument(
        "--img_size", type=int, default=128, help="size of the images. Default: 128."
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        default="none",
        help="root directory of the data and stored models",
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="dbg",
        help="name of the experiment for easy access to the results",
    )

    parser.add_argument(
        "--segm_classes",
        type=int,
        default=2,
        help="number of classes for segmentation. Default: 2.",
    )

    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=False,
        help="whether to use pretrained model. Default: False.",
    )

    parser.add_argument(
        "--best_hp_file_name",
        type=str,
        required=False,
        default="",
        help="name of the file with the best hyperparameters.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        default="",
        help="name of the file with the best hyperparameters.",
    )

    args, unparsed = parser.parse_known_args()

    args.train_episodes_config = {
        "n_way": args.n_way_train,
        "min_ways": None,
        "max_ways": None,
        "k_shot": args.k_shot_train,
        "min_shots": args.min_shots_train,
        "max_shots": args.max_shots_train,
        "query_size": args.query_size_train,
    }
    args.test_episodes_config = {
        "n_way": args.n_way_eval,
        "min_ways": args.min_ways_eval,
        "max_ways": args.max_ways_eval,
        "k_shot": args.k_shot_eval,
        "min_shots": args.min_shots_eval,
        "max_shots": args.max_shots_eval,
        "query_size": args.query_size_eval,
    }
    args.backbone = args.backbone.lower()

    return args, unparsed


class CrossTaskFewShotLearningExperiment:
    def __init__(self, args):
        self.args = args

        if self.args.best_hp_file_name:
            with open(
                os.path.join("hpo_summaries", self.args.best_hp_file_name + ".json")
            ) as f:
                hpo_results_dict = json.load(f)
                selected_configuration = hpo_results_dict["best_hyperparameters"]

            # overwrite the configurations
            if self.args.model == "tfs":
                for key in selected_configuration:
                    TFS_CONF[key] = selected_configuration[key]
            elif self.args.model == "finetuning":
                for key in selected_configuration:
                    FT_CONF[key] = selected_configuration[key]
                self.args.test_lr = selected_configuration["lr"]
                self.args.test_opt = selected_configuration["opt_fn"]
            elif self.args.model == "proto_finetuning":
                for key in selected_configuration:
                    PROTO_FT_CONF[key] = selected_configuration[key]
                self.args.test_lr = selected_configuration["lr"]
                self.args.test_opt = selected_configuration["opt_fn"]
            elif self.args.model == "maml":
                for key in selected_configuration:
                    MAML_CONF[key] = selected_configuration[key]
            elif self.args.model == "protomaml":
                for key in selected_configuration:
                    PROTO_MAML_CONF[key] = selected_configuration[key]
            elif self.args.model == "metacurvature":
                for key in selected_configuration:
                    METACURVATURE_CONF[key] = selected_configuration[key]
            elif self.args.model == "protonet":
                for key in selected_configuration:
                    PROTO_CONF[key] = selected_configuration[key]
            elif self.args.model == "matchingnet":
                for key in selected_configuration:
                    MATCHING_CONF[key] = selected_configuration[key]
            elif self.args.model == "ddrr":
                for key in selected_configuration:
                    DDRR_CONF[key] = selected_configuration[key]

        # Define paths
        self.curr_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        if self.args.root_dir != "none":
            self.main_dir = self.args.root_dir
        else:
            self.main_dir = self.curr_dir

        self.res_dir = os.path.join(self.main_dir, "results")
        self.data_dir = os.path.join(self.main_dir, "data")

        # Initialization step
        self.create_dirs()
        self.set_seed()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = get_device(self.logs_path)
        self.gpu_info = get_torch_gpu_environment()
        self.clprint = lambda text: lprint(text, self.logs_path)
        self.clprint("\n".join(self.gpu_info))
        self.configure()

    def create_dirs(self):
        create_dir(self.res_dir)

        self.res_dir += "/cross_problem_fsl/"
        create_dir(self.res_dir)

        self.res_dir += (
            self.args.test_id_datasets.replace(",", "_")
            + "_"
            + self.args.test_od_datasets.replace(",", "_")
            + "/"
        )
        create_dir(self.res_dir)

        if self.args.n_way_eval is None:
            n_ways = "AnyWay"
        else:
            n_ways = f"N{self.args.n_way_eval}"
        if self.args.k_shot_eval is None:
            k_shots = "AnyShot"
        else:
            k_shots = f"k{self.args.k_shot_eval}"
        self.res_dir += f"{n_ways}{k_shots}Test{self.args.query_size_eval}/"
        create_dir(self.res_dir)

        self.res_dir += (
            self.args.model
            + f"_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            + "/"
        )
        create_dir(self.res_dir)
        self.logs_path = f"{self.res_dir}logs.txt"

    def set_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

    def configure(self):
        # Mapping from model names to configurations
        mod_to_conf = {
            "tfs": (TrainFromScratch, deepcopy(TFS_CONF)),
            "finetuning": (FineTuning, deepcopy(FT_CONF)),
            "proto_finetuning": (ProtoFineTuning, deepcopy(PROTO_FT_CONF)),
            "maml": (MAML, deepcopy(MAML_CONF)),
            "protomaml": (ProtoMAML, deepcopy(PROTO_MAML_CONF)),
            "metacurvature": (MetaCurvature, deepcopy(METACURVATURE_CONF)),
            "protonet": (PrototypicalNetwork, deepcopy(PROTO_CONF)),
            "ddrr": (DDRR, deepcopy(DDRR_CONF)),
        }

        # Get model constructor and config for the specified algorithm
        self.model_constr, self.conf = mod_to_conf[self.args.model]

        # Set configurations
        self.overwrite_conf(self.conf, "train_batch_size")
        self.overwrite_conf(self.conf, "test_batch_size")
        self.overwrite_conf(self.conf, "T")
        self.overwrite_conf(self.conf, "lr")
        self.overwrite_conf(self.conf, "meta_batch_size")
        self.overwrite_conf(self.conf, "freeze")
        self.overwrite_conf(self.conf, "base_lr")

        if self.args.test_opt is not None or self.args.test_lr is not None:
            self.overwrite_conf(self.conf, "test_opt")
            self.overwrite_conf(self.conf, "test_lr")
        self.args.opt_fn = self.conf["opt_fn"]

        # Make sure argument 'val_after' is specified when 'validate'=True
        if self.args.validate:
            assert not self.args.val_after is None, (
                "Please specify "
                + "val_after (number of episodes after which to perform "
                + "validation)"
            )

        # If using multi-step maml, perform gradient clipping with -5, +5
        if "T" in self.conf:
            if self.conf["T"] > 1 and (
                self.args.model == "maml"
                or self.args.model == "protomaml"
                or self.args.model == "metacurvature"
            ):
                self.conf["grad_clip"] = 5
            else:
                self.conf["grad_clip"] = None

        if self.args.T_test is None:
            self.conf["T_test"] = self.conf["T"]
        else:
            self.conf["T_test"] = self.args.T_test

        if self.args.T_val is None:
            self.conf["T_val"] = self.conf["T"]
        else:
            self.conf["T_val"] = self.args.T_val

        self.conf["dev"] = self.device
        backbone_name = "resnet"
        num_blocks = int(self.args.backbone.split(backbone_name)[1])
        self.conf["baselearner_fn"] = ResNet

        self.args.test_episodes_config["model"] = self.args.model
        self.args.test_episodes_config["training"] = False
        self.args.train_episodes_config["model"] = self.args.model
        self.args.train_episodes_config["training"] = True

        num_tasks = self.args.eval_iters if self.args.val_id_datasets else 0
        val_id_datasets, val_id_dataset_task_type_dict = create_datasets(
            self.args.val_id_datasets.split(","), self.data_dir
        )
        self.val_id_loader = DataLoaderCrossProblem(
            val_id_datasets, num_tasks, self.args.test_episodes_config
        )

        num_tasks = self.args.eval_iters if self.args.val_od_datasets else 0
        val_od_datasets, val_od_dataset_task_type_dict = create_datasets(
            self.args.val_od_datasets.split(","), self.data_dir
        )
        self.val_od_loader = DataLoaderCrossProblem(
            val_od_datasets, num_tasks, self.args.test_episodes_config
        )

        num_tasks = self.args.eval_iters if self.args.test_id_datasets else 0
        test_id_datasets, test_id_dataset_task_type_dict = create_datasets(
            self.args.test_id_datasets.split(","), self.data_dir
        )
        self.test_id_loader = DataLoaderCrossProblem(
            test_id_datasets, num_tasks, self.args.test_episodes_config, True
        )

        num_tasks = self.args.eval_iters if self.args.test_od_datasets else 0
        test_od_datasets, test_od_dataset_task_type_dict = create_datasets(
            self.args.test_od_datasets.split(","), self.data_dir
        )
        self.test_od_loader = DataLoaderCrossProblem(
            test_od_datasets, num_tasks, self.args.test_episodes_config, True
        )

        num_train_classes = self.args.n_way_train
        if self.args.model in ("tfs", "finetuning", "proto_finetuning"):
            self.args.batchmode = True
            (
                train_datasets,
                train_dataset_task_type_dict,
                weights,
            ) = create_datasets_task_type(
                self.args.train_datasets.split(","), self.data_dir
            )
            if "finetuning" in self.args.model:
                # find the number of training classes for classification
                if "classification" in train_datasets:
                    num_train_classes = len(
                        train_datasets["classification"].idx_per_label
                    )
            if "segmentation" in train_datasets:
                self.args.segm_classes = (
                    len(train_datasets["segmentation"].idx_per_label) + 1
                )  # one class for background
            train_datasets = list(train_datasets.values())
        else:
            self.args.batchmode = False
            train_datasets, train_dataset_task_type_dict = create_datasets(
                self.args.train_datasets.split(","), self.data_dir
            )
            weights = None
        self.train_loader = DataLoaderCrossProblem(
            train_datasets,
            self.args.train_iters,
            self.args.train_episodes_config,
            weights=weights,
        )

        self.dataset_task_type_dict = {}
        for dataset in train_dataset_task_type_dict:
            self.dataset_task_type_dict[dataset] = train_dataset_task_type_dict[dataset]
        for dataset in val_id_dataset_task_type_dict:
            self.dataset_task_type_dict[dataset] = val_id_dataset_task_type_dict[
                dataset
            ]
        for dataset in val_od_dataset_task_type_dict:
            self.dataset_task_type_dict[dataset] = val_od_dataset_task_type_dict[
                dataset
            ]
        for dataset in test_id_dataset_task_type_dict:
            self.dataset_task_type_dict[dataset] = test_id_dataset_task_type_dict[
                dataset
            ]
        for dataset in test_od_dataset_task_type_dict:
            self.dataset_task_type_dict[dataset] = test_od_dataset_task_type_dict[
                dataset
            ]

        self.conf["baselearner_args"] = {
            "num_blocks": num_blocks,
            "dev": self.device,
            "train_classes": num_train_classes,
            "criterion": nn.CrossEntropyLoss(),
            "img_size": self.args.img_size,
            "segm_classes": self.args.segm_classes,
            "pretrained": self.args.pretrained,
        }

        # Print the configuration for confirmation
        self.clprint("\n\n### ------------------------------------------ ###")
        self.clprint(f"Model: {self.args.model}")
        self.clprint(f"Training Datasets: {self.args.train_datasets}")
        self.clprint(f"In-Domain Validation Datasets: {self.args.val_id_datasets}")
        self.clprint(f"Out-Domain Validation Datasets: {self.args.val_od_datasets}")
        self.clprint(f"In-Domain Testing Datasets: {self.args.test_id_datasets}")
        self.clprint(f"Out-Domain Testing Datasets: {self.args.test_od_datasets}")
        self.clprint(f"Random Seed: {self.args.seed}")
        self.print_conf()
        self.clprint("### ------------------------------------------ ###\n")

        with open(f"{self.res_dir}config.pkl", "wb") as f:
            pickle.dump(self.conf, f)
            print(f"Stored config in {self.res_dir}config.pkl")

    def overwrite_conf(self, conf, arg_str):
        # If value provided in arguments, overwrite the config with it
        value = getattr(self.args, arg_str)
        if value is not None:
            conf[arg_str] = value
        else:
            if arg_str not in conf:
                conf[arg_str] = None
            else:
                setattr(self.args, arg_str, conf[arg_str])

    def cycle(self, iterable):
        while True:
            for x in iterable:
                yield x

    def print_conf(self):
        self.clprint(f"Configuration dump:")
        for k in self.conf.keys():
            self.clprint(f"\t{k} : {self.conf[k]}")

    def run_test_evaluation(self, model, test_loader):
        test_scores = {}

        for i, task in enumerate(test_loader):
            ttime = time.time()
            n_way = task.n_way
            k_shot = task.k_shot
            query_size = task.query_size
            data = task.data
            labels = task.labels
            support_size = n_way * k_shot

            # Process the labels according to the task type
            if task.task_type == "segmentation":
                labels = task.segmentations.squeeze(dim=1)
            elif task.task_type.startswith("regression"):
                if task.task_type in [
                    "regression_pose_animals",
                    "regression_pose_animals_syn",
                    "regression_mpii",
                ]:
                    labels = get_k_keypoints(
                        n_way * 5, task.regressions, task.task_type
                    )
                else:
                    labels = task.regressions
            else:
                labels = process_labels(n_way * (k_shot + query_size), n_way)
            train_x, train_y, test_x, test_y = (
                data[:support_size],
                labels[:support_size],
                data[support_size:],
                labels[support_size:],
            )

            acc, loss_history, _, _ = model.evaluate(
                n_way,
                train_x,
                train_y,
                test_x,
                test_y,
                val=False,
                task_type=task.task_type,
            )
            test_loss = loss_history[-1]
            if task.dataset not in test_scores:
                test_scores[task.dataset] = []
            detailed_task_info = {
                "task_type": task.task_type,
                "dataset": task.dataset,
                "test_loss": test_loss,
                "score": float(acc),
                "method": self.args.model,
                "task_id": task.dataset
                + "_"
                + str(len(test_scores[task.dataset]) + 1).zfill(4),
            }
            test_scores[task.dataset].append(detailed_task_info)
            self.clprint(
                f"Iteration: {i+1}\t"
                + f"Test loss: {test_loss:.4f}\t"
                + f"Test score: {acc:.4f}\t"
                + f"Test time: {(time.time() - ttime):.2f} seconds"
            )

        return test_scores

    def load_state_from_disk(self, state_path):
        with open(state_path, "rb") as f:
            state_file = pickle.load(f)
        return state_file

    def run(self):
        # Create a placeholder model for cases when we do not test for in-domain or out-domain
        model = self.model_constr(**self.conf)

        seeds = [random.randint(0, 100000) for _ in range(self.args.runs)]
        print(f"Run seeds: {seeds}")

        results_dict = {}
        total_test_time = 0.0
        for run in range(self.args.runs):
            stime = time.time()

            self.clprint("\n\n" + "-" * 40)
            self.clprint(f"[*] Starting run {run} with seed {seeds[run]}")

            torch.manual_seed(seeds[run])
            model = self.model_constr(**self.conf)

            test_id_generator = iter(self.test_id_loader.generator(seeds[run]))
            test_od_generator = iter(self.test_od_loader.generator(seeds[run]))

            # Set seed and next test seed to ensure test diversity
            self.set_seed()

            test_time_start = time.time()

            # Load state
            best_state = self.load_state_from_disk(self.args.model_path)
            print(f"loading best state from {self.args.model_path}")
            model.load_state(best_state)
            self.clprint("\n[*] In-domain evaluation...")
            test_id_scores = self.run_test_evaluation(model, test_id_generator)
            self.clprint("\n[*] Out-domain evaluation...")
            test_od_scores = self.run_test_evaluation(model, test_od_generator)
            total_test_time += time.time() - test_time_start

            results_dict[run] = {
                "test_id_scores": {},
                "test_od_scores": {},
                "total_test_time": total_test_time,
                "method": self.args.model,
            }
            self.clprint(f"\nRun {run} done")
            for test_results, test_scores_name in zip(
                [test_id_scores, test_od_scores], ["test_id_scores", "test_od_scores"]
            ):
                self.clprint("\n" + test_scores_name)
                for dataset in test_results:
                    if self.dataset_task_type_dict[dataset] == "segmentation":
                        score_name = "mIoU"
                    elif self.dataset_task_type_dict[dataset] in [
                        "regression_pose_animals",
                        "regression_pose_animals_syn",
                        "regression_mpii",
                    ]:
                        score_name = "PCK"
                    elif self.dataset_task_type_dict[dataset] in [
                        "regression_distractor"
                    ]:
                        score_name = "euclidean distance"
                    elif self.dataset_task_type_dict[dataset] in [
                        "regression_pascal1d"
                    ]:
                        score_name = "degree error"
                    else:
                        score_name = "acc"

                    # task specific score
                    test_scores = [e["score"] for e in test_results[dataset]]
                    r, ts_mean, ts_median = (
                        str(run),
                        np.mean(test_scores),
                        np.median(test_scores),
                    )
                    ts_lb, _ = st.t.interval(
                        alpha=0.95,
                        df=len(test_scores) - 1,
                        loc=np.mean(test_scores),
                        scale=st.sem(test_scores),
                    )
                    ts_conf_interval = np.mean(test_scores) - ts_lb

                    # loss for all tasks
                    test_losses = [e["test_loss"] for e in test_results[dataset]]
                    r, tl_mean, tl_median = (
                        str(run),
                        np.mean(test_losses),
                        np.median(test_losses),
                    )
                    tl_lb, _ = st.t.interval(
                        alpha=0.95,
                        df=len(test_losses) - 1,
                        loc=np.mean(test_losses),
                        scale=st.sem(test_losses),
                    )
                    tl_conf_interval = np.mean(test_losses) - tl_lb

                    self.clprint(
                        f"\nDataset {dataset}, score_mean {score_name}: {ts_mean:.4f}, "
                        + f"score_median {score_name}: {ts_median:.4f}, score_95ci: {ts_conf_interval:.4f} "
                        + f"\nDataset {dataset},  loss: {tl_mean:.4f}, "
                        + f"median loss: {tl_median:.4f}, 95ci loss: {tl_conf_interval:.4f} "
                    )

                    test_dataset_results_dict = {
                        "test_score_name": score_name,
                        "test_score_mean": ts_mean,
                        "test_score_median": ts_median,
                        "test_score_conf_interval": ts_conf_interval,
                        "test_loss_mean": tl_mean,
                        "test_loss_median": tl_median,
                        "test_loss_conf_interval": tl_conf_interval,
                        "detailed_info": test_results[dataset],
                    }
                    results_dict[run][test_scores_name][
                        dataset
                    ] = test_dataset_results_dict

            used_time = time.time() - stime
            results_dict[run]["used_time"] = used_time
            self.clprint(f"\nTime(s): {used_time:.2f}")
            self.clprint("-" * 40)

        # Store all results into a file
        with open(
            os.path.join("summaries", self.args.experiment_name + ".json"), "w"
        ) as f:
            json.dump(results_dict, f)


if __name__ == "__main__":
    # Get args
    args, unparsed = parse_arguments()

    # If there is still some unparsed argument, raise error
    if len(unparsed) != 0:
        raise ValueError(f"Argument {unparsed} not recognized")

    # Initialize experiment object
    experiment = CrossTaskFewShotLearningExperiment(args)

    # Run experiment
    experiment.run()
