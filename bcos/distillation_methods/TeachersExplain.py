import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import resize
from torch.nn.functional import relu
from contextlib import nullcontext
from bcos.experiments.utils.config_utils import ExpMethod

class TeachersExplainCriterion(torch.nn.Module):
    #TODO DOCUMENT THIS
    def __init__(self,  config):
        super(TeachersExplainCriterion, self).__init__()

        # loss w.r.t GT labels and its coef
        self.gt_logit_criterion = config['gt_logit_criterion']
        self.gt_logit_loss_coef = config['gt_logit_loss_coef']

        # Teacher model and its overall loss coef
        self.teacher_coef = config['teacher_loss_coef']

        # loss w.r.t Teacher's logits and its coef
        self.teacher_logit_criterion = config['teacher_logit_criterion'] # This should now do any reduction!
        
        self.teacher_logit_loss_coef = config['teacher_logit_loss_coef']
        self.teacher_activation = config['teacher_activation']
        self.student_activation = config['student_activation']
        self.teacher_T = config['teacher_logit_temperature']
        self.logit_loss_rescale = config['teacher_logit_loss_rescale'] # Usually (temperature ** 2)

        # loss w.r.t Teacher's explanations and its coef
        self.exp_distill = config['explanation_distillation']
        self.explanation_method = config['explanation_method']
        self.exp_loss_coef = config['explanation_loss_coef']

        self.exp_of = config['exp_of']
        
        # Experimental Params
        self.guided_step_interval = config.get('guided_step_interval', 1) # To be read by outside


    def forward(self, model_dict, targets, teacher_dict, images, skip_exp=False, **kwargs):
        #TODO DOCUMENT THIS
        r"""
            Args:
                model_dict: Tensor (N, M), for M classes and N samples
                targets: Tensor of (N)
                teacher_dict: dict of logits snf explanations from offline/online teacher.
                It's assumed that the dict's items match the set explanation_method (each method works with specific keys).
                images: Tensor (N, C, H, W)
                skip_exp: Bool [default=False]. Allows to skip explanation evaluation (usually used for faster eval() steps)
            Returns:
                Tuple ( Scalar Loss Tensor, teacher's top-1 class predictions, Dict of different loss values for logging)
        """
        student_logits = model_dict['output']
        ######### Prepare Teacher Logits #########
        teacher_logits = teacher_dict['output'] # (N, num_teachers=1, num_classes)
        b_size, num_teachers = teacher_logits.shape[:2]

        teacher_selection_mask = model_dict['teacher_selection_mask']
        is_ensemble_distillation = num_teachers != 1 or torch.any(teacher_selection_mask != 0)
        assert not is_ensemble_distillation, 'Ensemble Mode not Implemented yet!'

        ######### COMPUTE EXPLANATION LOSS #########
        if (not self.exp_distill) or skip_exp:
            exp_loss, exp_loss_value = 0.0, None

        else:
            _exp_method = self.explanation_method.to_key()
            student_maps = model_dict[_exp_method] # (N, C, H, W)
            teacher_maps = teacher_dict[_exp_method] # (N, num_teachers=1, C, H, W)

            exp_loss = self._explanation_loss(pred_maps=student_maps, target_maps=teacher_maps[:, 0])
            exp_loss = exp_loss.mean()
            
            exp_loss_value = exp_loss.item()

        ########## Compute Logit Losses ##########

        ### Teacher Logit Loss
        if not is_ensemble_distillation: 
            teacher_logit_loss = self.teacher_logit_criterion(
                                    self.student_activation(student_logits, T=self.teacher_T), 
                                    self.teacher_activation(teacher_logits[:, 0], T=self.teacher_T)
                                ).sum(dim=-1).mean(dim=0) \
                            * self.logit_loss_rescale(self.teacher_T)
        else:
            raise NotImplementedError

        teacher0_preds = teacher_logits[:, 0].argmax(dim=1)
        
        ############ GT Logit Loss ############
        gt_logit_loss = self.gt_logit_criterion(student_logits, target=targets)

        ############ Put together the overall Loss #############
        loss = self.gt_logit_loss_coef * gt_logit_loss + \
            self.teacher_coef * (self.exp_loss_coef * exp_loss \
                                + self.teacher_logit_loss_coef * teacher_logit_loss)

        ret_dict =  dict(
            loss=loss.item(),
            logit_loss=teacher_logit_loss.item(),
            gt_logit_loss=gt_logit_loss.item(),
            teacher_logit_loss=teacher_logit_loss.item()
        )
        
        if exp_loss_value is not None: #Might be None because we haven't done explanation guidance
            ret_dict['exp_loss'] = exp_loss_value
        
        return loss, teacher0_preds, ret_dict

    def _explanation_loss(self, pred_maps, target_maps):
        #TODO DOCUMENT THIS
        """
        Computes the explanation loss between given and expected explanations.
        The loss is weighted-averaged over the batch based on given coefs. The loss is cosine similarity,
        which can be adjusted to L2 distance of [assumed] normalized maps by enabling self.adjustL2
        Args:
            pred_maps: Tensor (N, C, H, W),
            target_maps: Tensor (N, C, H, W),
            coefs: Tensor (N)
        Returns
            Scalar loss
        """
        sample_count = pred_maps.shape[0]
        sample_errors = 1 - F.cosine_similarity(pred_maps.view(sample_count, -1), target_maps.view(sample_count, -1))

        return sample_errors


def explanation_aware_forward(
        trainer, images, labels, teacher_dict=None,
        keep_graph=True, get_teacher_anyway=False, 
        skip_exp=False, get_student_anyway=False,
        no_teacher=False
    ):
    #TODO DOCUMENT THIS
    do_explain_student = (getattr(trainer, 'exp_distill', False) and (not skip_exp)) or get_student_anyway
    do_explain_teacher = trainer.livekd_training and (do_explain_student or get_teacher_anyway) and (not no_teacher)
    
    exp_method = getattr(trainer.criterion, 'explanation_method', None)
    exp_of = getattr(trainer.criterion, 'exp_of', None)
    if exp_method is None:
        assert get_teacher_anyway or get_student_anyway, 'Not clear why exp_method is None!'
        exp_of = 'top1'
        exp_method = ExpMethod.BCOS_WEIGHT if trainer.is_bcos else ExpMethod.GRAD_CAM_POS

    match exp_method:
        case ExpMethod.BCOS_WEIGHT:
            images.requires_grad = True; images.grad = None
            with trainer.model.explanation_mode():
                with trainer.teacher_ctx(): # Teacher EXP Mode
                    if trainer.livekd_training:
                        output, teacher_output = trainer(images)
                        if isinstance(output, dict):
                            output = output['output']
                        if isinstance(teacher_output, dict):
                            teacher_output = teacher_output['output']
                        if teacher_output.ndim == 2:
                            teacher_output = teacher_output.unsqueeze(1)
                    else:
                        output = trainer(images)
                        if isinstance(output, dict):
                            output = output['output']
                        teacher_output = teacher_dict['output']

                    teacher_selection_mask, exp_classes = select_classes_and_teachers(labels, exp_of, teacher_output)

                    with trainer.sync_ctx(): # create model_dict
                        images.grad = None; weight = None
                        if do_explain_student:
                            logits_to_explain = output[range(len(images)), exp_classes]
                            logits_to_explain.sum().backward(
                                inputs=[images], create_graph=keep_graph, retain_graph=keep_graph,
                            )
                            weight = images.grad
                        model_dict = dict(output=output, weight=weight, teacher_selection_mask=teacher_selection_mask)

                        if trainer.livekd_training: # Also create teacher_dict
                            images.grad = None; weight = None
                            if do_explain_teacher: 
                                logits_to_explain = teacher_output[range(len(images)), 0, exp_classes]
                                logits_to_explain.sum().backward(
                                    inputs=[images], create_graph=False, retain_graph=False,
                                )
                                weight = images.grad.detach()
                                if weight.ndim == 4:
                                    weight = weight.unsqueeze(1) # As if it's coming from multiple teachers
                            teacher_dict = dict(output=teacher_output.detach(), weight=weight)

                        images.grad = None # Otherwise cyclic-reference -> Memory Leak
        
        case ExpMethod.GRAD_CAM_POS:
            trainer.model.also_return_features=True 

            if trainer.livekd_training:
                trainer.teacher_model.also_return_features=True
                out_dict, t_out_dict = trainer(images)
                output, features = out_dict['output'], out_dict['features']
                t_output, t_features = t_out_dict['output'], t_out_dict['features']
                if t_output.ndim == 2:
                    t_output = t_output.unsqueeze(1) # as if it's coming from multiple teachers
            else:
                out_dict = trainer(images)
                output, features = out_dict['output'], out_dict['features']
                t_output = teacher_dict['output']

            teacher_selection_mask, exp_classes = select_classes_and_teachers(labels, exp_of, t_output)

            with trainer.sync_ctx():
                features.grad = None
                grad_cam_resized = None
                if do_explain_student:
                    logits_to_explain = output[range(len(images)), exp_classes]
                    logits_to_explain.sum().backward(
                        inputs=[features], create_graph=keep_graph, retain_graph=keep_graph,
                    )
                    grad_cam = (features.grad.mean((2,3), keepdim=True) * features).sum(1, keepdim=True)
                    grad_cam = relu(grad_cam)
                    grad_cam_resized = resize(grad_cam, (images.shape[-2], images.shape[-1]))
                model_dict = dict(output=output, grad_cam=grad_cam_resized, teacher_selection_mask=teacher_selection_mask)
                features.grad = None

                if trainer.livekd_training: # Also create teacher_dict
                    t_features.grad = None; grad_cam_resized = None
                    if do_explain_teacher: 
                        logits_to_explain = t_output[range(len(images)), 0, exp_classes]
                        logits_to_explain.sum().backward(
                            inputs=[t_features], create_graph=False, retain_graph=False,
                        )
                        grad_cam = (t_features.grad.mean((2,3), keepdim=True) * t_features).sum(1, keepdim=True).detach()
                        grad_cam = relu(grad_cam)
                        grad_cam_resized = resize(grad_cam, (images.shape[-2], images.shape[-1]))
                        if grad_cam_resized.ndim == 4:
                            grad_cam_resized = grad_cam_resized.unsqueeze(1) # As if it's coming from multiple teachers!
                    teacher_dict = dict(output=t_output.detach(), grad_cam=grad_cam_resized)
                    t_features.grad = None
                    trainer.teacher_model.also_return_features = False
            trainer.model.also_return_features = False

        case _:
            raise NotImplementedError

    return model_dict, teacher_dict

def select_classes_and_teachers(labels, exp_of, teacher_output):
    # teacher_output : [N, num_teachers, num_classes]
    assert teacher_output.shape[1] == 1, 'Ensemble mode is disabled for now!'
    if exp_of == 'top1':
        teacher_output = teacher_output[:, 0]
        _, top1_classes = torch.max(teacher_output, dim=-1) # [N] -> conf, pred indices
        teacher_selection_mask = torch.zeros_like(top1_classes)
        exp_classes = top1_classes
    else:
        raise NotImplementedError

    return teacher_selection_mask, exp_classes

def distillation_trainstep(trainer, batch, batch_idx):
    if trainer.ffkd_training:
        # Batch is a dict with precomputed logits and explanations
        images, labels = batch.pop('image'), batch.pop('target')
        teacher_dict = batch # Pre-computed teacher's logits and explanations
    else:
        images, labels = batch
        teacher_dict = None # To be overwritten in exp_aware_fwd() if trainer.livekd_training.

    skip_explaining_batch = (batch_idx % trainer.guided_step_interval != 0)
    model_dict, teacher_dict = explanation_aware_forward(trainer,
                    images, labels, teacher_dict=teacher_dict, keep_graph=True,
                    skip_exp = skip_explaining_batch)
    loss, teacher_preds, all_losses_dict = trainer.criterion(model_dict=model_dict, targets=labels,
                                            teacher_dict=teacher_dict, images=images,
                                            skip_exp=skip_explaining_batch)
    outputs = model_dict['output']

    return loss, outputs, labels, teacher_preds, all_losses_dict 

def distillation_evalstep(trainer, batch, batch_idx):
    if trainer.ffkd_training:
        # Batch is a dict with precomputed logits and explanations
        images, labels = batch.pop('image'), batch.pop('target')
        # The remainings pre-computed teacher's logits and explanations
        teacher_dict = batch 
    else:
        images, labels = batch
        teacher_dict = None # For now. Will be overwritten in exp_aware_fwd() if trainer.livekd_training is True.

    skip_explanations = ((trainer.trainer.current_epoch+1) % trainer.full_eval_every) != 0
    if (not trainer.exp_distill) or skip_explanations:
        maybe_no_grad = torch.no_grad
    else:
        maybe_no_grad = nullcontext
        torch.set_grad_enabled(True)
    
    with maybe_no_grad():
        #exp_aware_fwd() is also careful not to ask for grads if it shouldn't
        model_dict, teacher_dict = explanation_aware_forward(trainer,
                    images, labels, teacher_dict=teacher_dict, keep_graph=False,
                    skip_exp=skip_explanations)

    loss, teacher_preds, all_losses_dict = trainer.test_criterion(model_dict=model_dict, targets=labels,
                                                    images=images, teacher_dict=teacher_dict,
                                                    skip_exp=skip_explanations)
    outputs = model_dict['output']

    return loss, outputs, labels, teacher_preds, all_losses_dict

def distillation_init(trainer):
    trainer.exp_distill = getattr(trainer.criterion, 'exp_distill', False)
    trainer.guided_step_interval = getattr(trainer.criterion, 'guided_step_interval', 1)
    trainer.maybe_no_grad_teacher = nullcontext
    if trainer.livekd_training and ((not trainer.exp_distill) or trainer.criterion.explanation_method == ExpMethod.GRAD_CAM_POS):
        trainer.teacher_model.no_grad_features = True
