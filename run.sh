CONDA=/BS/mparcham/nobackup/conda/bin/conda
CONDA_ENV_NAME=MyPie

export IMAGENET_PATH='/scratch/inf0/user/mparcham/ILSVRC2012/'
export VOC_PATH='/scratch/inf0/user/mparcham/VOC2007/'
export WATERBIRDS_PATH='/scratch/inf0/user/mparcham/WaterBirds100/'

# SEED 720 is usually used for debug runs.
SEEDS=(720) 
TEMPERATURES=(1.0) # Sweep over Logit Temperature for KlDiv Loss
EXP_COEFS=(1.0) # Sweep over Explanation loss coefficients

for SEED in ${SEEDS[@]} 
do
    for TEMPERATURE in ${TEMPERATURES[@]}
    do  
        for EXP_COEF in ${EXP_COEFS[@]}
        do  
            ##### T: B-cos D169 --> S: B-cos R18 Distilled on ImageNet (Full), Temperature = 1.0, e2KD Coef = 1.0
            # GPU=4; RUN_NAME="resnet_18-GtL0.0-TeL1.0-TeT${TEMPERATURE}-EXP${EXP_COEF}-MethodBcosWeight-ExpOftop1-seed=${SEED}"

            ##### T: B-cos D169 --> S: B-cos R18 Distilled on ImageNet (with 50 Shots per class only), Temperature = 1.0, e2KD Coef = 1.0
            # GPU=1; RUN_NAME="resnet_18_FS50ShotBS32Sed42-GtL0.0-TeL1.0-TeT${TEMPERATURE}-EXP${EXP_COEF}-MethodBcosWeight-ExpOftop1-seed=${SEED}"

            ##### T: B-cos D169 --> S: B-cos R18 Distilled on ImageNet (with 200 Shots per class only), Temperature = 1.0, e2KD Coef = 1.0
            # GPU=1; RUN_NAME="resnet_18_FS200ShotBS64Sed42-GtL0.0-TeL1.0-TeT${TEMPERATURE}-EXP${EXP_COEF}-MethodBcosWeight-ExpOftop1-seed=${SEED}"

            ##### T: B-cos D169 --> S: B-cos ViT Tiny Distilled on ImageNet, Temperature = 1.0, e2KD Coef = 1.0
            GPU=1; RUN_NAME="bcos_simple_vit_ti_patch16_224-GtL0.0-TeL1.0-TeT${TEMPERATURE}-EXP${EXP_COEF}-MethodBcosWeight-ExpOftop1-seed=${SEED}"


            echo "--> on ${GPU} GPUS Running ${RUN_NAME}"
            ######## Use this one for running locally

            ${CONDA} run -n ${CONDA_ENV_NAME} --no-capture-output python3 train.py \
                --dataset ImageNet --base_network final_live --experiment_name ${RUN_NAME} \
                --explanation_logging --explanation_logging_every_n_epochs 1 \
                --wandb_logger --wandb_project "Teaching-Extensions" --track_grad_norm \
                \



            ######## Use this for running on Slurm (with submitit API)

            # ${CONDA} run -n ${CONDA_ENV_NAME} --no-capture-output python3 run_with_submitit.py \
                # --gpus ${GPU} --nodes 1 --timeout 6.5 --job_name ${RUN_NAME} --partition "gpu22,gpu20" \
                # --dataset ImageNet --base_network final_live --experiment_name ${RUN_NAME} \
                # --explanation_logging --explanation_logging_every_n_epochs 1 \
                # --wandb_logger --wandb_project "Teaching-Extensions" --track_grad_norm \

        done
    done
done