#import warnings
#warnings.filterwarnings("error")

import hydra
import torchvision
import os
import wandb
import pytorch_lightning as pl
import random
import numpy as np
import cProfile
import pstats


from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.profilers import AdvancedProfiler, SimpleProfiler

from omegaconf import DictConfig, OmegaConf
from datamodule import DataModule, PredictionsDataModule
from nn_models.model import Model
from calculator import Calculator
from log import Logger
from compare import compare_predictions
from rl_model import RL_Model


def compare(cfg: DictConfig):
    print("Comparing: ")
    compare_predictions(cfg.predictions_path, cfg.models_to_compare, cfg.image_name)



def see(cfg: DictConfig):

    print("Look at :", cfg.dataset.name)
    raise Exception("See cmd not upd to date!")

    # Create Data Module
    data_module = DataModule(cfg.dataset, cfg.dataloader, classes_as_known=cfg.dataset.classes_names)

    data_module.display_val_batch(shuffle=False)
    data_module.display_test_batch()

def set_random_seed(seed: int = 42):
    import torch # Here because it make a bug when using parrallel sweeper (I really do not known why...) :https://blog.csdn.net/m0_38052500/article/details/122132442
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you use multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def init_logger(cfg, name=None):

    # Create Logger
    logger = None
    if cfg.logger.wandb_on:
        config = OmegaConf.to_container(cfg, resolve=True)
        model_name = cfg.model.name
        if model_name == "faster_rcnn" and cfg.model.only_rpn:
            model_name = "rpn"
        if name == None:
            name = cfg.cmd + "_" + model_name + "_" + cfg.dataset.name
        logger = WandbLogger(name=name, project=cfg.logger.project_name, offline=cfg.logger.wandb_offline, config=config, job_type=cfg.cmd)
        wandb.init(**logger._wandb_init)

    return logger

def convert_lists_to_sets(config: DictConfig) -> DictConfig:
    """
    Recursively traverse the configuration and convert lists to sets.
    """
    for key, value in config.items():
        if isinstance(value, list):
            config[key] = set(value)
        elif isinstance(value, DictConfig):
            config[key] = convert_lists_to_sets(value)
    return config

def calcul_on_predictions(cfg: DictConfig):

    cfg = convert_lists_to_sets(cfg)


    logger = init_logger(cfg, name=cfg.cmd + "_" + cfg.dataset.dataset_name)

    data_directory = cfg.dataset.test_path  + "/" + cfg.dataset.split
    preload_predictions = cfg.dataset.preload_predictions #not "proposals" in cfg.models_name
    print(f"{preload_predictions = }")

    data_module = PredictionsDataModule(data_directory, set(cfg.models_name), cfg.dataset.max_size, batch_size=cfg.dataloader.batch_size, num_workers=cfg.dataloader.num_workers, preload_predictions=preload_predictions)
    print(f"Number of target outside limits is : {data_module.dataset.nb_target_remove_with_limit_area}, for {data_module.dataset.nb_target_boxes} target boxes, removing {(data_module.dataset.nb_target_remove_with_limit_area / data_module.dataset.nb_target_boxes) * 100} % of boxes")
    print(f"Number of target outside limits is : {data_module.dataset.nb_target_remove_with_limit_ratio}, for {data_module.dataset.nb_target_boxes} target boxes, removing {(data_module.dataset.nb_target_remove_with_limit_ratio / data_module.dataset.nb_target_boxes) * 100} % of boxes")
    calculator = Calculator(cfg.models_name, cfg.print_model_names, cfg.metrics, cfg.sort_prediction, set(cfg.dataset.classes_as_known), log_randoms=True, image_path=cfg.dataset.img_path + "/images/", is_log=cfg.is_log)

    #profiler= AdvancedProfiler(dirpath=".", filename="perf_logs")
    #profiler= SimpleProfiler(dirpath=".", filename="perf_logs")

    print("Precision is :", cfg.precision_float)

    trainer = pl.Trainer(accelerator='gpu',
                         logger=logger,
                         #profiler=profiler,
                         precision=cfg.precision_float,
                         )


    print("End Initialisation, Starting operation")

    profile = False
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    trainer.test(calculator, data_module)

    if profile:
        profiler.disable()
        output_file = "profiling_results.prof"
        profiler.dump_stats(output_file)

        stats = pstats.Stats(output_file)
        stats.sort_stats('tottime').print_stats(100)


    # Return value we want optuna to optimise
    to_optimise = trainer.logged_metrics["max_coverage_result"] 

    print("End Test")

    wandb.finish()

    return to_optimise



@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):


    if cfg.debug:
        print(OmegaConf.to_yaml(cfg))
        print("Dataset config : ", cfg.dataset)

    skip_train_load = False
    use_custom_scores = False

    if cfg.cmd == "see":
        return see(cfg)

    elif cfg.cmd == "compare":
        return compare(cfg)

    elif cfg.cmd == "test":
        skip_train_load = not cfg.test_on_train
        cfg.dataloader.shuffle_in_training = False

    elif cfg.cmd == "calcul_on_predictions":
        return calcul_on_predictions(cfg)

    elif cfg.cmd == "scores":
        use_custom_scores = True

    elif cfg.cmd != "train":
        raise Exception("[ERROR] The cmd " + cfg.cmd + " is not hundle, we have train, test, see and compare.")

    # Set the seed
    import torch # Here because it make a bug when using parrallel sweeper (I really do not known why...) :https://blog.csdn.net/m0_38052500/article/details/122132442
    set_random_seed(42)
    torch.set_float32_matmul_precision('high')
    torch.multiprocessing.set_sharing_strategy('file_system')
        

    # Create Logger
    wandb.finish()
    logger = init_logger(cfg)

    #Get correct model transform for dataset
    transform = None
    if cfg.model.pretrained:
        if cfg.model.name == "faster_rcnn":
            transform = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.transforms()

        elif cfg.model.name == "fcos":
            transform = torchvision.models.detection.FCOS_ResNet50_FPN_Weights.COCO_V1.transforms()
            print("Using fcos pretrained transform")

        elif cfg.model.name == "retina_net":
            transform = torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1.transforms()

        elif cfg.model.name == "yolov8":
            transform=None

        else:
            raise Exception("Model no hundle !")

    if cfg.dataset.name == "bdd100k" and cfg.model.ai_oro:
        print("\nForcing bdd100k to be with semantic segmentation !!!\n")
        cfg.dataset.semantics = True

    # Create Datasets
    data_module = DataModule(cfg.dataset,
                             cfg.dataloader,
                             cfg.model.classes_names_pred,
                             cfg.model.save_predictions_path + cfg.model.name,
                             skip_train_load=skip_train_load,
                             collate_fn=cfg.model.collate_fn,
                             transform=transform)

    if cfg.debug:
        print("CLASSES NAMES PREDICTED BY MODEL :", cfg.model.classes_names_pred)
        print("classes as known :", data_module.datasets.get_classes_as_known())
        print("classes as unknown :", data_module.datasets.get_classes_as_unknown())
        print("classes as background :", data_module.datasets.get_classes_as_background())
        print("data_module.datasets.semantic_segmentation_class_id_label : ", data_module.datasets.semantic_segmentation_class_id_label)

    cfg.model.save_predictions_path += cfg.dataset.name

    # Create Model
    model = Model(cfg.model,
                  cfg.metrics,
                  data_module.datasets.get_classes_as_known(),
                  classes_names_pred=cfg.model.classes_names_pred,
                  classes_names_gt=cfg.dataset.classes_names,  #data_module.datasets.merged_classes_names,              #cfg.dataset.classes_names,
                  classes_names_merged=data_module.datasets.merged_classes_names,
                  batch_size=cfg.dataloader.batch_size,
                  show_image=cfg.show_image,
                  use_custom_scores=use_custom_scores,
                  semantic_segmentation_class_id_label=data_module.datasets.semantic_segmentation_class_id_label)


    model.max_epochs = cfg.max_epochs

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    rpn = ""
    if cfg.model.prediction_from_rpn:
        rpn = "_rpn"
    type_of_model = "_"
    if not cfg.model.load:
        type_of_model = "_basic"
    elif cfg.model.ai_iou:
        type_of_model = "_iou"
    elif cfg.model.ai_oro:
        type_of_model = "_oro"
    else:
        type_of_model = "_?????"
    model.metrics_module.plt_name = cfg.dataset.name + "_" + cfg.model.name + rpn + type_of_model + "_v2_test"
    model.train_metrics_module.plt_name = cfg.dataset.name + "_" + cfg.model.name + rpn + type_of_model + "_v2_test"

    callbacks = [lr_monitor]
    if cfg.early_stopping:
        patience = max(2, int(cfg.max_epochs * 0.1))
        print("Early stopping on val coverage with patience at ", patience)
        early_stopping = EarlyStopping(monitor=cfg.early_stopping_monitor, mode="max", patience=patience) #TODO put in config
        callbacks.append(early_stopping)

    print("Precision is :", cfg.precision_float)

    print("\n Init done \n Starting operations :")



    # Create Trainer
    trainer = pl.Trainer(accelerator='gpu',
                         logger=logger,
                         max_epochs=cfg.max_epochs,
                         log_every_n_steps=cfg.logger.log_every_n_steps,
                         #accumulate_grad_batches=cfg.accumulate_gradient_nb_batch,
                         precision=cfg.precision_float,
                         check_val_every_n_epoch=cfg.check_val_every_n_epoch,
                         #callbacks=[EarlyStopping(monitor="val_map_known", mode="max")],
                         callbacks=callbacks,
                         num_sanity_val_steps=0
                         )
                         

    # Save config into save directory
    with open(model.save_path + "/config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    if cfg.cmd == "train":
        print("\nStart training")
        print("trainning set size :", len(data_module.dataset_train), "\n")
        model.dataset_split_name = "train"

        profile = False
        if profile:
            profiler = cProfile.Profile()
            profiler.enable()

        trainer.fit(model=model, datamodule=data_module)

        if profile:
            profiler.disable()
            output_file = "profiling_results.prof"
            profiler.dump_stats(output_file)

            stats = pstats.Stats(output_file)
            stats.sort_stats('tottime').print_stats(100)

        print("\nEnd training\n")


    # Test on all datasets depending of the config
    if cfg.test_on_train:
        print("Start testing on training dataset")
        model.dataset_split_name = "train"
        trainer.test(model=model, dataloaders=data_module.train_dataloader())

    if cfg.test_on_val:
        model.dataset_split_name = "val"
        print("Start testing on val dataset")
        trainer.test(model=model, dataloaders=data_module.val_dataloader())

    if cfg.test_on_test:
        model.dataset_split_name = "test"
        print("Start testing on test dataset")
        trainer.test(model=model, dataloaders=data_module.test_dataloader())

    wandb.finish()

if __name__ == "__main__":

    torchvision.disable_beta_transforms_warning()
    wandb.finish()
    main()
