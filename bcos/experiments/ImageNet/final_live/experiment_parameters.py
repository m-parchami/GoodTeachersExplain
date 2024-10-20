import math  # noqa
import torch.nn.functional as F

from bcos.data.presets import (
    ImageNetClassificationPresetEval,
    ImageNetClassificationPresetTrain,
)
from bcos.experiments.utils import (
    create_configs_with_different_criterion_params, ExpMethod,
    configs_cli,
    create_configs_with_different_seeds,
    update_config,
    safe_replace,
)
from bcos.modules import norms, DetachableGNLayerNorm2d
from bcos.modules.losses import (
    BinaryCrossEntropyLoss,
    UniformOffLabelsBCEWithLogitsLoss,
)
from bcos.distillation_methods import (
    TeachersExplain as TE, 
)
from torch.nn import KLDivLoss, CrossEntropyLoss, Identity
from bcos.optim import LRSchedulerFactory, OptimizerFactory

__all__ = ["CONFIGS"]

NUM_CLASSES = 1_000

DEFAULT_BATCH_SIZE = 64  # per GPU! * 4 = 256 effective
DEFAULT_NUM_EPOCHS = 200
DEFAULT_CROP_SIZE = 224

BCOS_OPTIMIZER = OptimizerFactory(name="Adam", lr=1e-3)
BASELINE_OPTIMIZER = OptimizerFactory(name="AdamW", lr=0.1, weight_decay=1e-4)
DEFAULT_LR_SCHEDULE = LRSchedulerFactory(
    name="cosineannealinglr",
    epochs=DEFAULT_NUM_EPOCHS,
    warmup_method="linear",
    warmup_epochs=5,
    warmup_decay=0.01,
)

EVAL_INTERVAL=1

DEFAULTS = dict(
    # Normal KD          # Fixed Teacher
    livekd_training=True, ffkd_training=False, 
    
    ## Fewshot Setting
    fewshot_training=False,
    train_samples_per_class=0, # Number of shots per class. Zero means all samples
    shot_seed=42, # Seed for the few shots.

    data=dict(
        val_split_at = 50, # 50 shots per class from the training data as validation set
        batch_size=DEFAULT_BATCH_SIZE,
        cache_dataset='shm',
        num_workers=32,
        num_classes=NUM_CLASSES,

        # Can be set for faster DataModule Creation
        # image_folder_dump_path='./stuff/pkls/IMN-train-Folder.pkl',
        # val_split_pkl = './stuff/pkls/GetCut-IMN-val.pkl',
    ),
    model=dict(
        args=dict(
            num_classes=NUM_CLASSES,
        )
    ),
    optimizer=None, # See Below
    lr_scheduler=DEFAULT_LR_SCHEDULE,
    trainer=dict(
        max_epochs=DEFAULT_NUM_EPOCHS,
        check_val_every_n_epoch=EVAL_INTERVAL,
    ),
    use_agc=True,
    criterion=None,
    test_criterion=None,

    # See Below
    criterion_config=None,
    test_criterion_config=None,
    criterion_module=None,
    distillation_trainstep=None,
    distillation_evalstep=None,
    distillation_init=None,
)


fewshot_dict = {
    nshots: (
        # Just a name tag
        f'FS{nshots}ShotBS{batch_size}Sed{shot_seed}',
        # Configs to be updated
        dict( 
            train_samples_per_class=nshots,
            fewshot_training=True,
            data=dict(
                batch_size=batch_size,
                val_split_at=50,
                
                # For faster collection of the shots:
                # fewshot_indices_pkl=f'./stuff/pkls/fewshot-fit-IMN-{nshots}-{shot_seed}',
            ),
            shot_seed=shot_seed,

            # Also update epoch-based configs (LR scheduling and the Trainer)
            lr_scheduler=LRSchedulerFactory(
                name="cosineannealinglr",
                epochs=epochs,
                warmup_method="linear",
                warmup_epochs=5,
                warmup_decay=0.01,
            ),
            trainer=dict(
                max_epochs=epochs,
                check_val_every_n_epoch=1,
            ),                
        )
    )
    for nshots, batch_size, epochs in [
        (50, 32, 250),
        (200, 64, 125),
    ]    
    for shot_seed in [42,]
}
# helper
def update_default(new_config):
    return update_config(DEFAULTS, new_config)


bcos_resnet = { # B-cos ResNet Student configs
    f"resnet_{depth}": update_default(
        dict(
            model=dict(
                is_bcos=True,
                name=f"resnet{depth}",
                args=dict(
                    norm_layer=norms.NoBias(norms.BatchNormUncentered2d),  # bnu-linear,
                    logit_bias=-math.log(NUM_CLASSES - 1),
                ),
                bcos_args=dict(
                    b=2,
                    max_out=1,
                ),
            ),
            optimizer=BCOS_OPTIMIZER,
            data=dict(
                train_transform=ImageNetClassificationPresetTrain(
                    crop_size=DEFAULT_CROP_SIZE,
                    is_bcos=True,
                ),
                test_transform=ImageNetClassificationPresetEval(
                    crop_size=DEFAULT_CROP_SIZE,
                    is_bcos=True,
                ),
            ),
        )
    )
    for depth in [18]
}
# -------------------------------------------------------------------------

# This TE config is re-used (with minor update()s ) for both B-cos and STD networks below.
DEFAULT_TE_CRITERION = dict(
    teacher_loss_coef = 1.0, # Deprecated - Always 1
    teacher_logit_criterion = KLDivLoss(reduction="none", log_target=False),

    # Logit loss configs
    gt_logit_loss_coef = 1.0,
    teacher_logit_loss_coef = 1.0,
    teacher_activation = lambda t, T: F.softmax(t/T, dim=1),
    student_activation = lambda s, T: F.log_softmax(s/T, dim=1), #Must be log space bcs of KLDivLoss
    teacher_logit_temperature = 1.0,
    teacher_logit_loss_rescale = lambda T: T**2,

    # Explanation usage configs
    explanation_distillation = True,
    explanation_loss_coef = 1.0,
    exp_of='top1',
)


############### B-cos DenseNet-169 Distillation Configs ###############
bcos_resnets_TE = {
    name: update_config(old_c,
        dict(
            criterion_config=update_config(
                DEFAULT_TE_CRITERION,
                dict(
                    gt_logit_criterion=UniformOffLabelsBCEWithLogitsLoss(), 
                    explanation_method=ExpMethod.BCOS_WEIGHT,
                    live_teacher_dataset='ImageNet',
                    live_teacher_base_network='bcos_final',
                    live_teacher_experiment_name='densenet_169',
                    live_teacher_ckpt_name='densenet_169-7037ee0604.pth',
                )
            ),
            test_criterion_same_as_train=True,
            criterion_module=TE.TeachersExplainCriterion,
            distillation_trainstep=TE.distillation_trainstep,
            distillation_evalstep=TE.distillation_evalstep,
            distillation_init=TE.distillation_init,
        )
    )
    for name, old_c in bcos_resnet.items()
}

#### Can be used for faster distillation. It applies the e2KD loss every K batches!
bcos_resnets_TE_efficient_guidance = {
    f"{name}_GuidanceEvery{guided_step_interval}Steps":  update_config(
        old_c,
        dict(
            criterion_config = dict(
                guided_step_interval=guided_step_interval
            )
        ),
    )
    for name, old_c in bcos_resnets_TE.items()
    for guided_step_interval in [16]
}

bcos_resnets_TE_fewshot = {
    f"{name}_{name_tag}": \
        update_config(old_c, config_update_dict)
    for name, old_c in bcos_resnets_TE.items()
    for nshot, (name_tag, config_update_dict) in fewshot_dict.items()
}



############### B-cos ResNet-34 Distillation Configs ###############
bcos_resnets_R34_TE = {
    f"{name}_R34Teacher": update_config(
        old_c,
        dict(
            criterion_config=dict(
                live_teacher_base_network='bcos_final',
                live_teacher_experiment_name='resnet_34',
                live_teacher_ckpt_name='resnet_34-a63425a03e.pth',
            ),
            test_criterion_same_as_train=True,
        )
    )
    for name, old_c in bcos_resnets_TE.items()
}

bcos_resnets_R34_TE_fewshot = {
    f"{name}_{name_tag}":  \
        update_config(old_c,config_update_dict)
    for name, old_c in bcos_resnets_R34_TE.items()
    for nshot, (name_tag, config_update_dict) in fewshot_dict.items()
}

############### B-cos ViT Configs ###############
bcos_vits_TE = {
    f"bcos_{name}": update_default(
        dict(
            model=dict(
                is_bcos=True,
                name=name,
                args=dict(
                    # DIFFERENT FROM RESNETS
                    norm_layer=norms.NoBias(norms.DetachableLayerNorm),
                    act_layer=Identity,
                    channels=6,
                    norm2d_layer=norms.NoBias(DetachableGNLayerNorm2d),
                ),
                logit_bias=-math.log(NUM_CLASSES - 1), # DIFFERENT FROM RESNETS (not in args!)
                bcos_args=dict(
                    b=2,
                    max_out=1,
                ),
            ),
            criterion_config=update_config(DEFAULT_TE_CRITERION,
                dict(
                    gt_logit_criterion = UniformOffLabelsBCEWithLogitsLoss(),
                    live_teacher_dataset='ImageNet',
                    live_teacher_base_network='bcos_final',
                    live_teacher_experiment_name='densenet_169',
                    live_teacher_ckpt_name='densenet_169-7037ee0604.pth',
                    explanation_method = ExpMethod.BCOS_WEIGHT,
                )
            ),
            test_criterion_config=update_config(DEFAULT_TE_CRITERION,
                dict(
                    gt_logit_criterion = BinaryCrossEntropyLoss(),
                    live_teacher_dataset='ImageNet',
                    live_teacher_base_network='bcos_final',
                    live_teacher_experiment_name='densenet_169',
                    live_teacher_ckpt_name='densenet_169-7037ee0604.pth',
                    explanation_method = ExpMethod.BCOS_WEIGHT,
                )
            ),
            criterion_module=TE.TeachersExplainCriterion,
            distillation_trainstep=TE.distillation_trainstep,
            distillation_evalstep=TE.distillation_evalstep,
            distillation_init=TE.distillation_init,
            optimizer=BCOS_OPTIMIZER,
            lr_scheduler = LRSchedulerFactory( # DIFFERENT FROM RESNETS
                name="cosineannealinglr",
                epochs=150,
                warmup_method="linear",
                warmup_steps=10_000,
                interval="step",
                warmup_decay=0.01,
            ),
            trainer=dict(
                max_epochs=150,
            ),
            data=dict(
                train_transform=ImageNetClassificationPresetTrain(
                    crop_size=DEFAULT_CROP_SIZE,
                    is_bcos=True,
                    auto_augment_policy="ra", # DIFFERENT FROM Resnets
                    ra_magnitude=10, # DIFFERENT FROM Resnets
                ),
                test_transform=ImageNetClassificationPresetEval(
                    crop_size=DEFAULT_CROP_SIZE,
                    is_bcos=True,
                ),
            ),
        )
    )
    for name in [
        # "vitc_ti_patch1_14",
        "simple_vit_ti_patch16_224",
    ]
}

#  Can be used for faster distillation with e2KD. Applied e2KD loss every K steps!
bcos_vits_TE_efficient = {
    f"{name}_GuidanceEvery{guided_step_interval}Steps": update_config(
        old_c,
        dict(
            criterion_config=dict(
                guided_step_interval=guided_step_interval
            ),
            test_criterion_config=dict(
                guided_step_interval=guided_step_interval
            ),
        )
    )
    for name, old_c in bcos_vits_TE.items()
    for guided_step_interval in [4]
}

BCOS_CONFIGS = dict()
BCOS_CONFIGS.update(bcos_resnets_TE)
BCOS_CONFIGS.update(bcos_resnets_TE_fewshot)

# BCOS_CONFIGS.update(bcos_resnets_R34_TE)
# BCOS_CONFIGS.update(bcos_resnets_R34_TE_fewshot)

BCOS_CONFIGS.update(bcos_vits_TE)
# BCOS_CONFIGS.update(bcos_resnets_TE_efficient_guidance)
# BCOS_CONFIGS.update(bcos_vits_TE_efficient)
#
BCOS_CONFIGS.update(
    create_configs_with_different_criterion_params(BCOS_CONFIGS, 
            params_dict=dict( # Don't change the order (will change order in the config names!)
                gt_logit_loss_coef = [0.0],
                teacher_logit_loss_coef = [1.0],
                teacher_logit_temperature = [1.0, 5.0],
                explanation_loss_coef = [0.0, 0.2, 1.0, 5.0, 10.0],
                explanation_method=[ExpMethod.BCOS_WEIGHT],
                exp_of=['top1']
            ),
            precomputed_paths_dict={
                ExpMethod.BCOS_WEIGHT: None,
            }
    )
)
#

bcos_resnets_fixed_TE = {
    f"{name}_FixedTeaching": update_config(old_c,
        dict(
            livekd_training=False, ffkd_training=True,
            data = dict(
                rescale_maps_to_img=True,
                teacher_intact=False,
            )
        )
    )
    for name, old_c in bcos_resnets_TE.items()
}

bcos_vit_fixed_TE = {
    f"{name}_FixedTeaching": update_config(old_c,
        dict(
            livekd_training=False, ffkd_training=True,
            data = dict(
                num_workers=24,
                rescale_maps_to_img=True,
                teacher_intact=False,
                train_transform=ImageNetClassificationPresetTrain(
                    crop_size=DEFAULT_CROP_SIZE,
                    is_bcos=True,
                    auto_augment_policy=None, # DIFFERENT FROM other ViT experiments
                ),
                test_transform=ImageNetClassificationPresetEval(
                    crop_size=DEFAULT_CROP_SIZE,
                    is_bcos=True,
                ),
            )
        )
    )
    for name, old_c in bcos_vits_TE.items()
}

#  Can be used for faster distillation with e2KD. Applied e2KD loss every K steps!
bcos_vit_fixed_TE_efficient_guidance = {
    f"{name}_GuidanceEvery{guided_step_interval}Steps":  update_config(
        old_c,
        dict(
            criterion_config = dict(
                guided_step_interval=guided_step_interval
            )
        ),
    )
    for name, old_c in bcos_vit_fixed_TE.items()
    for guided_step_interval in [
        16
    ]
}

bcos_resnets_fixed_TE_fewshot = {
    f"{name}_{name_tag}":  \
        update_config(old_c, config_update_dict)
    for name, old_c in bcos_resnets_fixed_TE.items()
    for nshot, (name_tag, config_update_dict) in fewshot_dict.items()
}


FIXED_BCOS_CONFIGS = dict()
# FIXED_BCOS_CONFIGS.update(bcos_resnets_fixed_TE)
# FIXED_BCOS_CONFIGS.update(bcos_resnets_fixed_TE_fewshot)
# FIXED_BCOS_CONFIGS.update(bcos_vit_fixed_TE)
# FIXED_BCOS_CONFIGS.update(bcos_vit_fixed_TE_efficient_guidance)
BCOS_CONFIGS.update(
    create_configs_with_different_criterion_params(FIXED_BCOS_CONFIGS, 
            params_dict=dict(
                gt_logit_loss_coef = [0.0],
                teacher_logit_loss_coef = [1.0],
                teacher_logit_temperature = [1.0, 5.0],
                explanation_loss_coef = [0.0, 0.2, 1.0, 5.0, 10.0],
                explanation_method=[ExpMethod.BCOS_WEIGHT],
                exp_of=['top1']
            ),
            precomputed_paths_dict={
                ExpMethod.BCOS_WEIGHT:\
                    '/scratch/inf0/user/mparcham/precomputed-submission-weights/ImageNet/bcos_final/densenet_169/',
            }
    )
)

BCOS_CONFIGS.update(create_configs_with_different_seeds(BCOS_CONFIGS, seeds=[550, 720]))



# -------------------------------------------------------------------------
old_setting_dict = dict( 
    optimizer=OptimizerFactory(
        name="SGD", lr=0.1,
        momentum=0.9, weight_decay=1e-4,
    ),
    lr_scheduler=LRSchedulerFactory(
        name="steplr", step_size=30, gamma=0.1,
    ),
    trainer=dict(
        max_epochs=100,
    ),
) # This config is only used in supplement for reproducing prior work.



standard_resnets = {
    f"std_resnet_{depth}": update_default(
        dict(
            model=dict(
                is_bcos=False,
                name=f"std_resnet{depth}",
            ),
            data=dict(
                train_transform=ImageNetClassificationPresetTrain(
                    crop_size=DEFAULT_CROP_SIZE,
                    is_bcos=False,
                ),
                test_transform=ImageNetClassificationPresetEval(
                    crop_size=DEFAULT_CROP_SIZE,
                    is_bcos=False,
                ),
            ),
            optimizer=BASELINE_OPTIMIZER,
            use_agc=False,
        )
    )
    for depth in [18]
}

standard_resnets_TE = {
    name: update_config(old_c, 
        dict(
            criterion_config=update_config(DEFAULT_TE_CRITERION,
                dict(
                    gt_logit_criterion = CrossEntropyLoss(), 
                    explanation_method = ExpMethod.GRAD_CAM_POS,
                    live_teacher_dataset='ImageNet',
                    live_teacher_base_network='baseline',
                    live_teacher_experiment_name=None, # See Below
                    live_teacher_ckpt_name=None,
                )
            ),
            test_criterion_same_as_train=True,
            criterion_module=TE.TeachersExplainCriterion,
            distillation_trainstep=TE.distillation_trainstep,
            distillation_evalstep=TE.distillation_evalstep,
            distillation_init=TE.distillation_init,          
        )
    )
    for name, old_c in standard_resnets.items()
}

standard_resnets_TE_LRs = {
    f"{name}_Lr{lr}": update_config(old_c,
            dict(
                optimizer=OptimizerFactory(name="AdamW", lr=lr, weight_decay=1e-4),
                trainer=dict(
                    gradient_clip_val=1.0,
                )
            )
        )
    for name, old_c in standard_resnets_TE.items()
    for lr in [0.01,]
}

standard_resnets_R34Teacher_TE = {
    f"{name}_R34Teacher": update_config(old_c, 
        dict(
            criterion_config=dict(
                live_teacher_experiment_name='std_resnet_34',
                live_teacher_ckpt_name='resnet34-b627a593.pth',
            ),
            test_criterion_same_as_train=True,
        )
    )
    for name, old_c in standard_resnets_TE.items()
}

standard_resnets_R34Teacher_TE_LRs = {
    f"{name}_Lr{lr}": update_config(old_c,
            dict(
                optimizer=OptimizerFactory(name="AdamW", lr=lr, weight_decay=1e-4),
                trainer=dict(
                    gradient_clip_val=1.0,
                ),
            )
        )
    for name, old_c in standard_resnets_R34Teacher_TE.items()
    for lr in [0.01,]
}

standard_resnets_R34Teacher_TE_LRs_fewshot = {
    f'{name}_{name_tag}' : \
        update_config(old_c, config_update_dict)
    for name, old_c in standard_resnets_R34Teacher_TE_LRs.items()
    for nshot, (name_tag, config_update_dict) in fewshot_dict.items()
}

STANDARD_CONFIGS = dict()


# R-34 Teacher
# STANDARD_CONFIGS.update(standard_resnets_R34Teacher_TE_LRs)
# STANDARD_CONFIGS.update(standard_resnets_R34Teacher_TE_LRs_fewshot)
# STANDARD_CONFIGS.update(standard_resnets_R34Teacher_TE_oldSetting)

STANDARD_CONFIGS.update(
    create_configs_with_different_criterion_params(STANDARD_CONFIGS, 
            params_dict=dict( # Don't change the order (will change order in the config names!)
                gt_logit_loss_coef = [0.0],
                teacher_logit_loss_coef = [1.0, 0.5],
                teacher_logit_temperature = [1.0, 5.0],
                explanation_loss_coef = [10.0, 5.0, 1.0, 0.0],
                explanation_method=[ExpMethod.GRAD_CAM_POS],
                exp_of=['top1']
            ),
            precomputed_paths_dict={
                ExpMethod.GRAD_CAM_POS: None,
            }
    )
)



# STANDARD_CONFIGS.update(standard_resnets_LRs_fewshot)

STANDARD_CONFIGS.update(create_configs_with_different_seeds(STANDARD_CONFIGS, seeds=[550]))


CONFIGS = dict()
CONFIGS.update(BCOS_CONFIGS)
# CONFIGS.update(STANDARD_CONFIGS)

if __name__ == "__main__":
    configs_cli(CONFIGS)
