import pdb

import mmint_utils
import numpy as np
import torch
from neural_contact_fields.data.tool_dataset import ToolDataset
from neural_contact_fields.undcf_decoder_only.generation import get_surface_loss_fn
from neural_contact_fields.undcf_decoder_only.models.virdo_undcf import VirdoUNDCF
from neural_contact_fields.training import BaseTrainer
import os
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import Dataset
import neural_contact_fields.loss as ncf_losses


class Trainer(BaseTrainer):

    def __init__(self, cfg, model: VirdoUNDCF, device):
        super().__init__(cfg, model, device)

        self.model: VirdoUNDCF = model

    ##########################################################################
    #  Pretraining loop                                                      #
    ##########################################################################

    def pretrain(self, pretrain_dataset: Dataset):
        pass

    ##########################################################################
    #  Main training loop                                                    #
    ##########################################################################

    def train(self, train_dataset: ToolDataset, validation_dataset: ToolDataset):
        # Shorthands:
        out_dir = self.cfg['training']['out_dir']
        lr = self.cfg['training']['learning_rate']
        max_epochs = self.cfg['training']['epochs']
        epochs_per_save = self.cfg['training']['epochs_per_save']
        self.train_loss_weights = self.cfg['training']['loss_weights']  # TODO: Better way to set this?

        # Output + vis directory
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        logger = SummaryWriter(os.path.join(out_dir, 'logs/train'))

        # Dump config to output directory.
        mmint_utils.dump_cfg(os.path.join(out_dir, 'config.yaml'), self.cfg)

        # Get optimizer (TODO: Parameterize?)
        optimizer_latent = optim.Adam(self.model.trial_code.parameters(), lr=lr)
        optimizer_model = optim.Adam([
            {"params": self.model.deformation_model.parameters()},
            {"params": self.model.contact_model.parameters()},
            {"params": self.model.wrench_encoder.parameters()},
        ], lr=lr)

        # Load model + optimizer if a partially trained copy of it exists.
        epoch_it, it = self.load_partial_train_model(
            {"model": self.model, "optimizer_latent": optimizer_latent, "optimizer_model": optimizer_model}, out_dir,
            "model.pt")

        # Training loop
        while True:
            epoch_it += 1

            if epoch_it > max_epochs:
                print("Backing up and stopping training. Reached max epochs.")
                save_dict = {
                    'model': self.model.state_dict(),
                    'optimizer_latent': optimizer_latent.state_dict(),
                    'optimizer_model': optimizer_model.state_dict(),
                    'epoch_it': epoch_it,
                    'it': it,
                }
                torch.save(save_dict, os.path.join(out_dir, 'model.pt'))
                break

            loss = None

            trial_idcs = np.arange(len(train_dataset))
            np.random.shuffle(trial_idcs)
            trial_idcs = torch.from_numpy(trial_idcs).to(self.device)

            for trial_idx in trial_idcs:
                it += 1

                # For this training, we use just a single example per run.
                batch = train_dataset[trial_idx]
                loss = self.train_step(batch, it, optimizer_latent, optimizer_model, logger)

            print('[Epoch %02d] it=%03d, loss=%.4f'
                  % (epoch_it, it, loss))

            if epoch_it % epochs_per_save == 0:
                save_dict = {
                    'model': self.model.state_dict(),
                    'optimizer_latent': optimizer_latent.state_dict(),
                    'optimizer_model': optimizer_model.state_dict(),
                    'epoch_it': epoch_it,
                    'it': it,
                }
                torch.save(save_dict, os.path.join(out_dir, 'model.pt'))

    def train_step(self, data, it, optimizer_latent, optimizer_model, logger):
        """
        Perform training step. This wraps up the gradient calculation for convenience.

        Args:
        - data (dict): batch data dict
        - it (int): training iter
        - optimizer (torch.optim.Optimizer): optimizer used for training
        - logger: tensorboard logger being used.
        - compute_loss_fn (callable): loss function to call. Should return a loss dictionary with main
          loss at key "loss" and an out_dict
        """
        self.model.train()

        ###################################
        # First loss: update latent codes #
        ###################################
        optimizer_latent.zero_grad()
        optimizer_model.zero_grad()

        z_object, z_trial, z_wrench = self.compute_latent(data, it)
        latent_loss_dict, latent_out_dict = self.compute_latent_train_loss(data, z_object, z_trial, z_wrench)

        for k, v in latent_loss_dict.items():
            logger.add_scalar(k, v, it)

        latent_loss = latent_loss_dict['loss']
        latent_loss.backward()
        optimizer_latent.step()

        #####################################
        # Second loss: update model weights #
        #####################################
        optimizer_latent.zero_grad()
        optimizer_model.zero_grad()

        z_object, z_trial, z_wrench = self.compute_latent(data, it)
        model_loss_dict, model_out_dict = self.compute_model_train_loss(data, z_object, z_trial, z_wrench)

        for k, v in model_loss_dict.items():
            logger.add_scalar(k, v, it)

        model_loss = model_loss_dict['loss']
        model_loss.backward()
        optimizer_model.step()

        return latent_loss + model_loss

    def compute_latent(self, data, it):
        object_idx = data["object_idx"]
        trial_idx = data["trial_idx"]
        wrist_wrench = data["wrist_wrench"].float().unsqueeze(0)
        z_object, z_trial = self.model.encode_trial(object_idx, trial_idx)
        z_wrench = self.model.encode_wrench(wrist_wrench)

        return z_object, z_trial, z_wrench

    def compute_latent_train_loss(self, data, z_object, z_trial, z_wrench):
        partial_pointcloud = data["partial_pointcloud"].float().unsqueeze(0)

        # Run model forward.
        out_dict = self.model.forward(partial_pointcloud, z_trial, z_object, z_wrench)

        # Loss:
        loss_dict = dict()

        # SDF Loss: all partial point cloud points should lie on surface.
        sdf_loss = torch.mean(torch.abs(out_dict["sdf"]), dim=-1)
        loss_dict["partial_sdf_loss"] = sdf_loss

        # Latent embedding loss:
        embedding_loss = ncf_losses.l2_loss(out_dict["embedding"], squared=True)
        loss_dict["embedding_loss"] = embedding_loss

        # Calculate total weighted loss.
        loss = 0.0
        for loss_key in loss_dict.keys():
            loss += float(self.train_loss_weights[loss_key]) * loss_dict[loss_key]
        loss_dict["loss"] = loss

        return loss_dict, out_dict

    def compute_model_train_loss(self, data, z_object, z_trial, z_wrench):
        coords = data["query_point"].float().unsqueeze(0)
        gt_sdf = data["sdf"].float().unsqueeze(0)
        gt_in_contact = data["in_contact"].float().unsqueeze(0)
        nominal_coords = data["nominal_query_point"].float().unsqueeze(0)
        nominal_sdf = data["nominal_sdf"].float().unsqueeze(0)

        # Run model forward.
        out_dict = self.model.forward(coords, z_trial, z_object, z_wrench)

        # Loss:
        loss_dict = dict()

        # SDF Loss: How accurate are the SDF predictions at each query point.
        sdf_loss = ncf_losses.sdf_loss(out_dict["sdf"], gt_sdf)
        loss_dict["sdf_loss"] = sdf_loss

        # Chamfer distance loss.
        chamfer_loss = ncf_losses.surface_chamfer_loss(nominal_coords, nominal_sdf, gt_sdf, out_dict["nominal"])
        loss_dict["chamfer_loss"] = chamfer_loss

        # Loss on deformation field.
        def_loss = ncf_losses.l2_loss(out_dict["deform"], squared=True)
        loss_dict["def_loss"] = def_loss

        # Contact prediction loss.
        in_contact_dist = out_dict["in_contact_dist"]
        contact_loss = ncf_losses.heteroscedastic_bce(in_contact_dist, gt_in_contact)
        contact_loss = contact_loss[gt_sdf == 0.0].mean()
        loss_dict["contact_loss"] = contact_loss

        # Network regularization.
        reg_loss = self.model.regularization_loss(out_dict)
        loss_dict["reg_loss"] = reg_loss

        # Calculate total weighted loss.
        loss = 0.0
        for loss_key in loss_dict.keys():
            loss += float(self.train_loss_weights[loss_key]) * loss_dict[loss_key]
        loss_dict["loss"] = loss

        return loss_dict, out_dict
