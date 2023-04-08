import argparse
import os
import random

import mmint_utils
import numpy as np
import torch
import neural_contact_fields.metrics as ncf_metrics
from neural_contact_fields import config
from neural_contact_fields.utils import utils
from neural_contact_fields.utils.results_utils import load_gt_results, load_pred_results, print_results
from tqdm import trange, tqdm
import torchmetrics
import pytorch3d.loss


def calculate_metrics(dataset_cfg_fn: str, dataset_mode: str, out_dir: str, verbose: bool = False,
                      sample: bool = False):
    device = torch.device("cuda:0")

    # Load dataset.
    dataset_config = mmint_utils.load_cfg(dataset_cfg_fn)
    dataset = config.get_dataset(dataset_mode, dataset_config)
    num_trials = len(dataset)
    dataset_dir = dataset_config["data"][dataset_mode]["dataset_dir"]

    # Check if there are multiple runs.
    run_dirs = [f for f in os.listdir(out_dir) if "run_" in f]
    if len(run_dirs) == 0:
        run_dirs = ["./"]

    with tqdm(total=len(run_dirs) * num_trials) as pbar:
        for run_dir in run_dirs:
            run_out_dir = os.path.join(out_dir, run_dir)

            # Load specific ground truth results needed for evaluation.
            gt_meshes, gt_pointclouds, gt_contact_patches, gt_contact_labels, points_iou, gt_occ_iou = \
                load_gt_results(
                    dataset, dataset_dir, num_trials, device
                )

            # Load predicted results.
            pred_meshes, pred_pointclouds, pred_contact_patches, pred_contact_labels, pred_iou_labels, misc = \
                load_pred_results(run_out_dir, num_trials, device)

            # Calculate metrics.
            metrics_results = []
            for trial_idx in range(num_trials):
                metrics_dict = dict()

                # Evaluate meshes.
                if pred_meshes[trial_idx] is not None:
                    chamfer_dist = ncf_metrics.mesh_chamfer_distance(pred_meshes[trial_idx], gt_meshes[trial_idx],
                                                                     device=device,
                                                                     vis=verbose)
                    iou = ncf_metrics.mesh_iou(points_iou[trial_idx], gt_occ_iou[trial_idx], pred_meshes[trial_idx],
                                               device=device,
                                               vis=verbose)
                    metrics_dict.update({
                        "chamfer_distance": chamfer_dist.item() * 1e6,
                        "iou": iou.item(),
                    })

                # Evaluate pointclouds.
                if pred_pointclouds[trial_idx] is not None:
                    if sample:
                        pred_pc = utils.sample_pointcloud(pred_pointclouds[trial_idx], 10000)
                    gt_pc = utils.sample_pointcloud(gt_pointclouds[trial_idx], 10000)
                    chamfer_dist, _ = pytorch3d.loss.chamfer_distance(pred_pc.unsqueeze(0).float(),
                                                                      gt_pc.unsqueeze(0).float())

                    metrics_dict.update({
                        "chamfer_distance": chamfer_dist.item() * 1e6,
                    })

                # Evaluate contact patches.
                if pred_contact_patches[trial_idx] is not None:
                    # Sample each to 300 - makes evaluation of CD more fair.
                    if sample:
                        pred_pc = utils.sample_pointcloud(pred_contact_patches[trial_idx], 300)
                    else:
                        pred_pc = pred_contact_patches[trial_idx]
                    gt_pc = utils.sample_pointcloud(gt_contact_patches[trial_idx], 300)
                    patch_chamfer_dist, _ = pytorch3d.loss.chamfer_distance(
                        pred_pc.unsqueeze(0).float(),
                        gt_pc.unsqueeze(0).float())

                    metrics_dict.update({
                        "patch_chamfer_distance": patch_chamfer_dist.item() * 1e6,
                    })

                # Evaluate binary contact labels.
                if pred_contact_labels[trial_idx] is not None:
                    pred_contact_labels_trial = pred_contact_labels[trial_idx]["contact_labels"].float()
                    binary_accuracy = torchmetrics.functional.classification.binary_accuracy(pred_contact_labels_trial,
                                                                                             gt_contact_labels[
                                                                                                 trial_idx],
                                                                                             threshold=0.5)
                    precision = torchmetrics.functional.classification.binary_precision(pred_contact_labels_trial,
                                                                                        gt_contact_labels[trial_idx],
                                                                                        threshold=0.5)
                    recall = torchmetrics.functional.classification.binary_recall(pred_contact_labels_trial,
                                                                                  gt_contact_labels[trial_idx],
                                                                                  threshold=0.5)
                    f1 = torchmetrics.functional.classification.binary_f1_score(pred_contact_labels_trial,
                                                                                gt_contact_labels[trial_idx],
                                                                                threshold=0.5)
                    metrics_dict.update({
                        "binary_accuracy": binary_accuracy.item(),
                        "precision": precision.item(),
                        "recall": recall.item(),
                        "f1": f1.item(),
                    })

                if pred_iou_labels[trial_idx] is not None:
                    pred_iou_labels_trial = pred_iou_labels[trial_idx]["iou_labels"].float()
                    gt_iou_labels_trial = gt_occ_iou[trial_idx].float()

                    iou = torchmetrics.functional.classification.binary_jaccard_index(pred_iou_labels_trial,
                                                                                      gt_iou_labels_trial,
                                                                                      threshold=0.5)

                    metrics_dict.update({
                        "model_iou": iou.item(),
                    })

                if misc[trial_idx] is not None:
                    for key in ["mesh_gen_time", "latent_gen_time", "iters"]:
                        if key in misc[trial_idx]:
                            metrics_dict[key] = misc[trial_idx][key]

                metrics_results.append(metrics_dict)

                pbar.update()

            if verbose:
                print_results(metrics_results, os.path.dirname(run_out_dir))

            # Write all metrics to file.
            if run_out_dir is not None:
                mmint_utils.save_gzip_pickle(metrics_results, os.path.join(run_out_dir, "metrics.pkl.gzip"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate metrics on generated data.")
    parser.add_argument("dataset_cfg", type=str, help="Dataset configuration.")
    parser.add_argument("out_dir", type=str, help="Out directory where results are written to.")
    parser.add_argument("--mode", "-m", type=str, default="test", help="Dataset mode [train, val, test]")
    parser.add_argument("--verbose", "-v", dest='verbose', action='store_true', help='Be verbose.')
    parser.set_defaults(verbose=False)
    parser.add_argument("--sample", "-s", dest='sample', action='store_true',
                        help='Sample pointclouds to set number before evaluation.')
    parser.set_defaults(sample=False)
    args = parser.parse_args()

    # Seed for repeatability.
    torch.manual_seed(10)
    np.random.seed(10)
    random.seed(10)

    calculate_metrics(args.dataset_cfg, args.mode, args.out_dir, args.verbose, args.sample)
