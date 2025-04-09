import argparse
import pickle

import numpy as np
import torch

from pathlib import Path

from cmflib import cmf
from mdwc2025 import models
from mdwc2025.training import test_model, train_model
from torch.utils.data import DataLoader, random_split

WEIGHT_PATH = Path("/lustre/data/shao/cug_2025/pretrained_models")

def cmf_init(execution_stage, mlmd_file):
    metawriter = cmf.Cmf(
        filepath=mlmd_file, pipeline_name="CMF-SmartSim-2025", graph=True
    )

    context = metawriter.create_context(pipeline_stage="Model Retraining")

    execution = metawriter.create_execution(execution_stage)
    return metawriter


def main(args):

    metawriter = cmf_init(f"{args.model_arch}_retrainer", args.mlmd_file)

    # Log the input models

    # Open the base dataset
    print("Reading in base dataset")
    with open(args.base_data, "rb") as f:
        dataset = pickle.load(f)
    metawriter.log_dataset(args.base_data, "input")

    # Read in all the extra data
    print("Reading in extra datasets")
    for fname in args.extra_data:
        metawriter.log_dataset(fname, "input")
        with open(fname, "rb") as f:
            extra_ds = pickle.load(f)
            dataset.append(extra_ds.features, extra_ds.target)

    # Sample evenly across clusters
    idx = []
    clusters = dataset.clusters.predict(dataset.features)
    for i in range(dataset.clusters.labels_.max()+1):
        cluster_indices = np.argwhere(clusters == i).squeeze()
        idx.extend(np.random.choice(cluster_indices, 5000, replace=False))
    dataset.features = dataset.features[idx,:]
    dataset.target = dataset.target[idx]
    with open("training_dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)

    # Sample evenly from all clusters
    # Train/Test Split
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size  # Ensure exact match

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # DataLoaders
    val_loader = DataLoader(val_dataset, batch_size=131072, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=131072, shuffle=False)

    # Train and Save Model
    model = train_model(
        train_dataset,
        args.model_arch,
        val_loader,
        metawriter,
        device_spec=args.device,
        weights_file=WEIGHT_PATH / f"{args.model_arch}.pth",
        convergence_criterion=0.5,
        batch_size=131072,
        subset_data=False,
        num_epochs=100
    )
    metawriter.commit_metrics("training_metrics")
    metawriter.commit_metrics("validation_metrics")

    model_label = f"{args.model_arch}_retrained"
    model_path = model_label + ".pth"
    torch.save(model.state_dict(), model_path)

    # JIT-trace the model for inference
    model.eval()
    module = torch.jit.trace(model, torch.rand((1,4), device=args.device))
    jit_model_path = f"{model_label}_jit.pt"
    torch.jit.save(module, jit_model_path)

    # Run Test
    test_loss = test_model(model, test_loader, metawriter)
    metawriter.log_model(
        path=str(Path(jit_model_path).resolve()),
        event="output",
        model_framework="pytorch",
        model_type=args.model_arch,
        model_name=model_label,
        custom_properties={
            "test_loss": test_loss
        }
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mlmd_file", help="Path to the cmf database file for metadata tracking"
    )
    parser.add_argument(
        "model_arch",
        help="The architecture of the ML model to train",
        choices=["EKEResNet", "EKEBottleneckResNet"],
    )
    parser.add_argument(
        "base_data", help="File (pickle) which contains the base dataset"
    )
    parser.add_argument(
        "test_data", help="File (pickle) which contains the test dataset"
    )
    parser.add_argument(
        "extra_data", help="Files (pickle) which contain additional data", nargs="+"
    )
    parser.add_argument("--device", help="The device to use for training")
    args = parser.parse_args()
    print(args)
    main(args)
