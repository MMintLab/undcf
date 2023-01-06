from collections import defaultdict

import numpy as np
import torch
import tqdm
from neural_contact_fields.neural_contact_field.models.neural_contact_field import NeuralContactField
from torch import nn
from tqdm import trange
import torch.optim as optim
import neural_contact_fields.loss as ncf_losses
import torch.nn.functional as F


def infer_latent(model: NeuralContactField, trial_dict: dict, loss_weights: dict, device=None):
    model.eval()

    # Initialize latent code as noise.
    z_deform_ = nn.Embedding(1, model.z_deform_size, dtype=torch.float32).requires_grad_(True).to(device)
    torch.nn.init.normal_(z_deform_.weight, mean=0.0, std=0.1)
    optimizer = optim.Adam(z_deform_.parameters(), lr=2e-3)

    z_deform = z_deform_.weight
    for ep in range(1000):
        # Pull out relevant data.
        object_idx = torch.from_numpy(trial_dict["object_idx"]).to(device)
        coords = torch.from_numpy(trial_dict["query_point"]).to(device).float().unsqueeze(0)
        trial_idx = torch.from_numpy(trial_dict["trial_idx"]).to(device)
        gt_sdf = torch.from_numpy(trial_dict["sdf"]).to(device).float().unsqueeze(0)
        gt_normals = torch.from_numpy(trial_dict["normals"]).to(device).float().unsqueeze(0)
        gt_in_contact = torch.from_numpy(trial_dict["in_contact"]).to(device).float().unsqueeze(0)
        nominal_coords = torch.from_numpy(trial_dict["nominal_query_point"]).to(device).float().unsqueeze(0)
        nominal_sdf = torch.from_numpy(trial_dict["nominal_sdf"]).to(device).float().unsqueeze(0)

        # We assume we know the object code.
        z_object = model.encode_object(object_idx)

        # Predict with updated latents.
        pred_dict = model.forward(coords, z_deform, z_object)

        # Loss:
        loss_dict = dict()

        # SDF Loss: How accurate are the SDF predictions at each query point.
        sdf_loss = ncf_losses.sdf_loss(pred_dict["sdf"], gt_sdf)
        loss_dict["sdf_loss"] = sdf_loss

        # Normals loss: are the normals accurate.
        normals_loss = ncf_losses.surface_normal_loss(gt_sdf, gt_normals, pred_dict["normals"])
        loss_dict["normals_loss"] = normals_loss

        # Latent embedding loss: well-formed embedding.
        embedding_loss = ncf_losses.l2_loss(pred_dict["embedding"], squared=True)
        loss_dict["embedding_loss"] = embedding_loss

        # Loss on deformation field.
        def_loss = ncf_losses.l2_loss(pred_dict["deform"], squared=True)
        loss_dict["def_loss"] = def_loss

        # Contact prediction loss.
        contact_loss = F.binary_cross_entropy_with_logits(pred_dict["in_contact_logits"][gt_sdf == 0.0],
                                                          gt_in_contact[gt_sdf == 0.0])
        loss_dict["contact_loss"] = contact_loss

        # Chamfer distance loss.
        chamfer_loss = ncf_losses.surface_chamfer_loss(nominal_coords, nominal_sdf, gt_sdf, pred_dict["nominal"])
        loss_dict["chamfer_loss"] = chamfer_loss

        # Calculate total weighted loss.
        loss = 0.0
        for loss_key in loss_dict.keys():
            loss += float(loss_weights[loss_key]) * loss_dict[loss_key]
        loss_dict["loss"] = loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tqdm.tqdm.write("Epoch: %d, Loss: %f" % (ep, loss.item()))

    # Predict with final latent.
    pred_dict = model.forward(coords, z_deform, z_object)
    return z_deform_, pred_dict


def infer_latent_from_surface(model, trial_dict, max_batch: int = 40 ** 3, device=None):
    model.eval()

    query_points = torch.from_numpy(trial_dict["query_point"]).to(device).float()
    sdf = torch.from_numpy(trial_dict["sdf"]).float().to(device)
    surface_query_points = sdf == 0.0
    num_samples = surface_query_points.shape[0]
    query_points = query_points[surface_query_points]

    # Initialize latent code as noise.
    latent_code = torch.zeros([1, 64], requires_grad=True, dtype=torch.float32, device=device)
    torch.nn.init.normal_(latent_code, mean=0.0, std=0.1)
    optimizer = optim.Adam([latent_code], lr=1e-3)

    for it in trange(100000):
        optimizer.zero_grad()

        head = 0
        num_iters = int(np.ceil(num_samples / max_batch))
        pred_dict_all = defaultdict(list)
        for iter_idx in range(num_iters):
            sample_subset = query_points[head: min(head + max_batch, num_samples)]
            latent_in = latent_code.repeat(sample_subset.shape[0], 1)

            pred_dict = model.latent_forward(latent_in, sample_subset)

            for k, v in pred_dict.items():
                pred_dict_all[k].append(v)

            head += max_batch

        # Apply surface loss.
        loss = torch.sum(torch.abs(pred_dict["sdf"]))

        if (it % 1000) == 0:
            print("Step %d: %f" % (it, loss))

        loss.backward()
        optimizer.step()

    return latent_code


def points_inference(model: NeuralContactField, trial_dict, device=None):
    model.eval()
    object_index = trial_dict["object_idx"]
    trial_index = trial_dict["trial_idx"]

    # Encode object idx/trial idx.
    z_object, z_trial = model.encode_trial(torch.from_numpy(object_index).to(device),
                                           torch.from_numpy(trial_index).to(device))

    # Get query points to sample.
    query_points = torch.from_numpy(trial_dict["query_point"]).to(device).float()
    pred_dict = model.forward(query_points.unsqueeze(0), z_trial, z_object)

    return pred_dict
