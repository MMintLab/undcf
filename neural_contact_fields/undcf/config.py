from neural_contact_fields.undcf.models.virdo_undcf import VirdoUNDCF
from neural_contact_fields.undcf.training import Trainer
from neural_contact_fields.undcf.generation import Generator
from neural_contact_fields.undcf.visualization import Visualizer


def get_model(cfg, dataset, device=None):
    model_cfg = cfg["model"]

    try:
        num_objects = dataset.get_num_objects()
        num_trials = dataset.get_num_trials()
    except:
        raise Exception("Training with unexpected dataset type: %s." % str(type(dataset)))

    model = VirdoUNDCF(num_objects, num_trials, model_cfg["z_object_size"], model_cfg["z_deform_size"],
                       model_cfg["z_wrench_size"], device=device)
    return model


def get_trainer(cfg, model, device=None):
    trainer = Trainer(cfg, model, device)
    return trainer


def get_generator(cfg, model, generation_cfg, device=None):
    generator = Generator(cfg, model, generation_cfg, device)
    return generator


def get_visualizer(cfg, model, device=None, visualizer_args=None):
    visualizer = Visualizer(cfg, model, device, visualizer_args)
    return visualizer
