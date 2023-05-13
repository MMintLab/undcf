import argparse
import random

import numpy as np
import torch
from tqdm import trange

from neural_contact_fields.utils.model_utils import load_dataset_from_config

from vedo import Plotter, Mesh, Points


def vis_dataset(dataset, offset: int = 0):
    print("Dataset size: %d" % len(dataset))

    for data_idx in trange(offset, len(dataset)):
        data_dict = dataset[data_idx]

        partial_pc = data_dict["partial_pointcloud"]
        contact_patch = data_dict["contact_patch"]
        mesh = dataset.get_example_mesh(data_idx)

        # Visualize.
        plt = Plotter()
        plt.at(0).show(
            Mesh([mesh.vertices, mesh.faces], c="grey", alpha=0.5),
            Points(partial_pc, c="black"),
            Points(contact_patch, c="red"),
        )
        plt.interactive().close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize dataset.")
    parser.add_argument("dataset_cfg", type=str, help="Path to dataset config.")
    parser.add_argument("--mode", "-m", type=str, default="test", help="Which dataset split to use [train, val, test].")
    parser.add_argument('--offset', type=int, default=0, help='Offset to start from.')
    args = parser.parse_args()

    _, dataset_ = load_dataset_from_config(args.dataset_cfg, dataset_mode=args.mode)
    vis_dataset(dataset_, offset=args.offset)