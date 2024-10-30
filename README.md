<p align="center">
<h1 align="center">
Good Teachers Explain: Explanation-Enhanced Knowledge Distillation
</h1>

<p align="center">
<a href="https://www.linkedin.com/in/amin-parchami"><strong>Amin Parchami-Araghi<sup>*</sup></strong></a>
·
<a href="https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/people/moritz-boehle/"><strong>Moritz Böhle<sup>*</sup></strong></a>
·
<a href="https://sukrutrao.github.io"><strong>Sukrut Rao<sup>*</sup></strong></a>
·
<a href="https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/people/bernt-schiele"><strong>Bernt Schiele</strong></a>
</p>
  
<h3 align="center">
<a href="https://arxiv.org/abs/2402.03119v2">ECCV 2024</a>
</h3>
</p>

<img src="https://github.com/m-parchami/GoodTeachersExplain/assets/56980571/c5914d4c-82da-491c-a961-c37d1fe1d6db">



# Overview of the Codebase

This codebase was adapted from [B-cos-v2 Codebase](https://github.com/B-cos/B-cos-v2). It is based on Pytorch Lightning and comes with several useful features such as Distributed Data Parallel, Slurm integration (via submitit), and WandB logging. I therefore changed the codebase to work for our distillation setting. That means, the overall codebase is not fully optimized for KD and if you are looking for a distillation codebase, I strongly suggest popular codebases such as [Torchdistill](https://github.com/yoshitomo-matsubara/torchdistill). Please also refer only to [B-cos-v2 Codebase](https://github.com/B-cos/B-cos-v2) for the most up-to-date implementation on B-cos Networks.


# Getting Started

After cloning the repo, please download the teacher checkpoints using the following: 
```bash
bash download_teachers.sh
```
Afterwards, refer to `run.sh` for sample training commands. You would need to adjust the dataset paths at beginning of the script.


# Important files
Most of the essential implementations of our method can be found under `bcos/distillation_methods/TeachersExplain.py`.

- **The Loss:** The `forward()` pass computes the losses for both logits and explanations.

- **Computing the Explanations:** The explanations are computed together with logits under `explanation_aware_forward()`.

- **Train and Eval Step:** There are three callbacks for distillation: `distillation_` (`trainstep()`, `evalstep()`, and `init()`). These are also defined under `bcos/distillation_methods/TeachersExplain.py` to be called by the Trainer class (as defined in `bcos/training/trainer.py`). During training or validation, the Trainer will call either `distillation_trainstep()` or `distillation_evaltep()` to process a batch and compute the loss. Their main difference is the gradient computation. `distillation_init()` is also called only once during initialization of the Trainer. 


# Configs for Experiments

Similar to [B-cos-v2 Codebase](https://github.com/B-cos/B-cos-v2), the configs are implemented as key-value pairs using standard distionaries. Our configs for ImageNet experiments can be found in `bcos/experiments/ImageNet/final_live/experiment_parameters.py`. 

In order to avoid repitition, the dictionaries may override each other. For example, a fewshot KD experiment, has the same exact config as a normal experiment, except for the changes for fewshot settings. 

Please note that the configs may generate hyperparameter combinations that were not actually tested in the paper. To see the hyperparameters that we have used for the paper, kindly refer to the Section C in our supplement.

# Citation
If you use our work or our codebase, please cite our work as follows:
```
@inproceedings{parchamiaraghi2024good,
      title         = {Good Teachers Explain: Explanation-Enhanced Knowledge Distillation}, 
      author        = {Amin Parchami-Araghi and Moritz Böhle and Sukrut Rao and Bernt Schiele},
      booktitle     = {European Conference on Computer Vision},
      year          = {2024},
}
```

# Pending Changes
- [ ] Adding Waterbirds Configs
- [ ] Adding Pascal VOC Configs
- [ ] Adding Method Documentation
- [ ] Adding reference for borrowed code snippets
