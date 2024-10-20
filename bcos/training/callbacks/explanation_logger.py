import copy
import io
import math
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
import pytorch_lightning.loggers as pl_loggers
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as transformsF
from torch.nn.functional import relu
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn

from contextlib import nullcontext

from  bcos.experiments.utils.config_utils import ExpMethod
from bcos.common import gradient_to_image
from bcos.distillation_methods.TeachersExplain import explanation_aware_forward

def to_numpy(tensor: "Union[torch.Tensor, np.ndarray]") -> "np.ndarray":
    if not isinstance(tensor, torch.Tensor):
        return tensor
    return tensor.detach().cpu().numpy()


# class that creates a cpu copy of the model
class ModelCopy(torch.nn.Module):
    def __init__(self, model: "torch.nn.Module", use_cpu: bool = True):
        super().__init__()
        self.device = (
            torch.device("cpu") if use_cpu else next(model.parameters()).device
        )
        self.model = copy.deepcopy(model).to(self.device)
        self.original_model = model  # ref to original model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def explanation_mode(self, *args, **kwargs):
        return self.model.explanation_mode(*args, **kwargs)

    def update(self):
        """Update the copy with the original model's parameters."""
        with torch.no_grad():
            # get original model's state dict and put tensors on given device
            state_dict = {
                k: v.to(self.device, non_blocking=True, copy=True)
                for k, v in self.original_model.state_dict().items()
            }
            self.model.load_state_dict(state_dict)

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad = False


# explanation logger
from plotting_utils import visualize_image_attr_custom
class ExplanationsLogger(pl_callbacks.Callback):
    def __init__(
        self,
        log_every_n_epochs: int = 1,
        idx: Optional[torch.Tensor] = None,
        max_imgs: int = 32,
    ):
        self.log_every_n_epochs = log_every_n_epochs
        self.max_imgs = max_imgs
        self.idx = idx

        self.imgs = None
        self.lbls = None

        self.has_bcos_model = False

        self.model: "Optional[ModelCopy]" = None

    @rank_zero_only
    def setup(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str
    ) -> None:
        if stage != "fit":
            return

        if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
            dataset = trainer.datamodule.eval_dataset
        else:
            dataset = trainer.val_dataloaders[0].dataset

        # we have a bcos model so we now we can safely do explanation log
        self.has_bcos_model = getattr(pl_module, 'is_bcos', False)

        max_imgs = self.max_imgs
        idx = self.idx
        if idx is None:
            g = torch.Generator().manual_seed(42)
            idx = torch.randint(high=len(dataset), size=(max_imgs,), generator=g)
        self.idx = idx[:max_imgs]

        imgs = []
        lbls = []
        teacher_exps = []
        self.is_ffkd = pl_module.ffkd_training
        self.is_livekd = pl_module.livekd_training
        self.has_teacher = self.is_ffkd or self.is_livekd
        if self.has_teacher:
            self.exp_method =  pl_module.config['criterion_config'].get(
                    'explanation_method', 
                    ExpMethod.BCOS_WEIGHT if self.has_bcos_model else ExpMethod.GRAD_CAM_POS # Default explanation methods
                )
            # We want to propagate teacher with Grad
            orig_teacher_grad_ctx = pl_module.maybe_no_grad_teacher
            pl_module.maybe_no_grad_teacher = nullcontext
        for i in idx:
            item = dataset[i]
            if self.is_ffkd:
                # In case item has some extra stuff; ignore those for now!
                img, lbl = item.get('image'), item.get('target')
                teacher_exps.append(item.get(self.exp_method.to_key()).squeeze())
            else:
                img, lbl = item[0], item[1]
                if self.is_livekd:
                    _, teacher_dict = explanation_aware_forward(
                        trainer=pl_module, images=img[None], labels=[lbl], 
                        teacher_dict=None, keep_graph=False, 
                        get_teacher_anyway=True, skip_exp=True
                    )
                    teacher_exps.append(teacher_dict.get(self.exp_method.to_key()).cpu().squeeze()) #(6, H, W) or (H, W)
            imgs.append(img.cpu())
            lbls.append(lbl if isinstance(lbl, int) or torch.numel(lbl) == 1 else lbl.argmax())
        self.imgs = torch.stack(imgs)
        if self.has_teacher:
            pl_module.maybe_no_grad_teacher = orig_teacher_grad_ctx
            self.teacher_exps = torch.stack(teacher_exps)

        self.lbls = torch.tensor(lbls)

        self.model = ModelCopy(pl_module.model, use_cpu=True)

    @rank_zero_only
    def on_validation_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        
        if trainer.sanity_checking:
            return

        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        # not initialized yet
        if self.model is None or self.imgs is None or self.lbls is None:
            return

        wandb_logger = None
        tensorboard_logger = None
        for logger in trainer.loggers:
            if isinstance(logger, pl_loggers.WandbLogger):
                wandb_logger = logger
            elif isinstance(logger, pl_loggers.TensorBoardLogger):
                tensorboard_logger = logger

        self.log_explanations(pl_module, wandb_logger, tensorboard_logger)

    def log_explanations(
        self,
        pl_module,
        wandb_logger: "Optional[pl_loggers.WandbLogger]" = None,
        tensorboard_logger: "Optional[pl_loggers.TensorBoardLogger]" = None,
        postfix="",
    ):
        if wandb_logger is None and tensorboard_logger is None:
            return

        results = self.get_results(pl_module)
        explanations = self.plot_explanations(results)
        expl_pil_img = self.fig_to_pil(explanations)

        namespace = f"explanations{postfix}"
        if wandb_logger:
            wandb_logger.log_image(f"Explanations/{namespace}", [expl_pil_img])

        if tensorboard_logger:
            writer = tensorboard_logger.experiment
            if hasattr(writer, "add_figure"):
                writer.add_figure(f"{namespace}/Explanations", explanations)
            else:
                writer.add_image(
                    f"{namespace}/Explanations", transformsF.to_tensor(expl_pil_img)
                )

        # figures are retained, hence manual clean up
        plt.close(explanations)

    def get_results(self, pl_module):
        # prepare things
        imgs = self.imgs
        lbls = self.lbls
        assert self.model is not None
        self.model.update()
        self.model.eval()
        self.model.freeze()  # just to be sure

        pl_module.model.eval()

        device = self.model.device

        # result dict
        
        explanation_ctx = self.model.explanation_mode if self.has_bcos_model else nullcontext
        results = []  # (img, lbl, pred, w)
        with torch.enable_grad(), explanation_ctx():
            # calculate expls
            for idx, (img, lbl) in enumerate(zip(imgs, lbls)):
                assert not img.requires_grad and img.grad is None
                # prep data
                lbl = lbl.item()
                if self.has_bcos_model:
                    img = img[None].to(device).requires_grad_()

                    # get predictions
                    out = self.model(img)
                    if isinstance(out, dict):
                        out = out['output']
                    pred = out.max(1)
                    pred.values.backward(inputs=[img])
                    pred = pred.indices.item()

                    # dynamic weights
                    grad = img.grad
                    exp = grad.detach().cpu().squeeze() # (6, H, W)
                else:
                    # Take GradCam Explanations
                    img = img[None].cuda().requires_grad_()
                    model_dict, _ = explanation_aware_forward(
                        trainer=pl_module, images=img, labels=[lbl], 
                        teacher_dict=None, keep_graph=False,
                        get_teacher_anyway=False, no_teacher=True, get_student_anyway=True
                    )
                    pred = model_dict['output'].argmax(dim=-1).item()
                    exp = model_dict[self.exp_method.to_key()].detach().cpu().squeeze() # (H, W)
                # add to results
                if not self.has_teacher:
                    result = (
                        img.detach().cpu().squeeze(),
                        lbl, pred,
                        exp,
                    )
                else:
                    result = (
                        img.detach().cpu().squeeze(),
                        lbl, pred,
                        exp,
                        self.teacher_exps[idx]
                    )
                results.append(result)

        return results

    def plot_explanations(self, results):
        N = len(results) * 2
        H, W = self.get_grid_size(N)

        if self.has_teacher:
            # Add a row per each result row
            H += math.ceil(N/W)

        fig = plt.figure(dpi=200)
        grid = ImageGrid(
            fig,
            111,  # similar to subplot(111)
            nrows_ncols=(H, W),
            axes_pad=0.0,  # is okay
        )

        for i, result in enumerate(results):
            if self.has_teacher:
                (img, lbl, pred, exp, teacher_exp) = result
            else:
                (img, lbl, pred, exp) = result
            
            x = i // W
            y = i - W * x
            factor = 2 if not self.has_teacher else 3
            img_ax = grid[factor * x * W + y]
            expl_ax = grid[(factor * x + 1) * W + y]
            if self.has_teacher: target_expl_ax = grid[(factor * x + 2) * W + y]
            
            if self.has_bcos_model:
                img_ax.imshow(to_numpy(img[:3].permute(1, 2, 0)))
                expl = to_numpy(gradient_to_image(img, exp))
                expl_ax.imshow(expl)
            else:
                img_ax.imshow(to_numpy(
                    img.permute(1, 2, 0) * torch.Tensor([0.229, 0.224, 0.225]).view(1, 1, 3) +\
                        torch.Tensor([0.485, 0.456, 0.406]).view(1, 1, 3))
                    )
                if exp.ndim == 3 and exp.shape[0] > 1:
                    exp = exp - exp.min()
                    exp = exp / exp.max()
                    expl = to_numpy(exp.moveaxis(0, -1))
                    expl_ax.imshow(expl)
                else:
                    expl = to_numpy(exp)
                    visualize_image_attr_custom(expl, sign="all", show_colorbar=False, 
                        plt_fig_axis=(fig, expl_ax), use_pyplot=False, scale_factor=np.percentile(expl, 95))

            if self.has_teacher:
                match self.exp_method:
                    case ExpMethod.BCOS_WEIGHT | ExpMethod.BCOS_CONTRIB:
                        target_expl = gradient_to_image(img, teacher_exp)
                        target_expl_ax.imshow(to_numpy(target_expl))
                    case _:
                        if teacher_exp.ndim == 3 and teacher_exp.shape[0] > 1:
                            teacher_exp = teacher_exp - teacher_exp.min()
                            teacher_exp = teacher_exp / teacher_exp.max()
                            target_expl = to_numpy(teacher_exp.moveaxis(0, -1))
                            target_expl_ax.imshow(target_expl)
                        else:
                            target_expl = teacher_exp
                            visualize_image_attr_custom(target_expl, sign="all", show_colorbar=False, 
                                plt_fig_axis=(fig, target_expl_ax), use_pyplot=False, scale_factor=np.percentile(target_expl, 95))
            # style
            for ax in (img_ax, expl_ax, target_expl_ax) if self.has_teacher else (img_ax, expl_ax):
                ax.set_xticks([])
                ax.set_yticks([])
            for spine in expl_ax.spines.values():
                spine.set_edgecolor("lime" if lbl == pred else "orange")
                spine.set_linestyle((0, (5, 3)) if lbl == pred else (0, (2, 3)))

        return fig

    @staticmethod
    def get_grid_size(n, ratio=2 / 3):
        """Returns grid size (H,W) with
        H/W approx= ratio and H*W>=n. and H=even.
        """
        x = int(math.sqrt(ratio * n))
        if x % 2 == 1:
            x += 1
        # doesn't have to be perfect just good enough
        y = int(math.ceil(n / x))
        return x, y

    @staticmethod
    def fig_to_pil(fig):
        # from https://stackoverflow.com/a/61754995/10614892
        buf = io.BytesIO()
        fig.savefig(buf, format="jpeg", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        img = Image.open(buf)
        return img

    @staticmethod
    def convert_image(pil_img):
        # TODO: if needed change
        return pil_img
