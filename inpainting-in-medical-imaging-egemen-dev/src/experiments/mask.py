from typing import Any, Dict

import numpy as np
from torch import optim
import torch
import torch.nn.functional as F
from torch import nn
import segmentation_models_pytorch as smp

from trainer import BaseTrainer
from trainer.callbacks.timer import TimerCallback
from trainer.typing import PBarType

from src.utils.visuals import (
    create_confusion_matrix_image,
    create_grid_grade,
    create_grid_shape,
)
from src.utils.losses import (
    JointLoss,
    DiceLoss,
    TargetArgmaxLossWrapper,
)
from src.utils.metrics import (
    calc_confusion_matrix,
    calc_dice_score,
    calc_precision,
    calc_recall,
)


class MaskExperiment(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.callbacks.append(TimerCallback())

        assert self.params is not None

        self.shape_enabled = self.params["shape_segmenter"]["enabled"]
        self.grade_enabled = self.params["grade_segmenter"]["enabled"]

        assert self.shape_enabled or self.grade_enabled

        if self.shape_enabled:
            cfg = self.params["shape_segmenter"]
            self.networks.add(
                id="shape",
                network=self.select_segmenter(cfg, 2, 2).to(self.device),
                additional_info=cfg,
            )
            self.optimizers.add(
                id="shape",
                optimizer=optim.Adam(
                    self.networks.get_parameters("shape"),
                    lr=cfg["optimizer"]["lr"],
                    betas=(cfg["optimizer"]["beta1"], cfg["optimizer"]["beta2"]),
                    weight_decay=cfg["optimizer"]["weight_decay"],
                ),
                additional_info=cfg["optimizer"],
            )
            self.shape_criterion = self.select_criterion(cfg["criterion"])
            if cfg["scheduler"]["enabled"]:
                self.schedulers.add(
                    id="shape",
                    scheduler=optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer=self.optimizers.get("shape"),
                        mode=cfg["scheduler"]["mode"],
                        factor=cfg["scheduler"]["factor"],
                        patience=cfg["scheduler"]["patience"],
                        verbose=True,
                    ),
                    metric_id=cfg["scheduler"]["metric_name"],
                    metric_category=cfg["scheduler"]["metric_category"],
                    metric_stage=cfg["scheduler"]["metric_stage"],
                    additional_info=cfg["scheduler"],
                )

        if self.grade_enabled:
            cfg = self.params["grade_segmenter"]
            self.networks.add(
                id="grade",
                network=self.select_segmenter(cfg, 2, 4).to(self.device),
                additional_info=cfg,
            )
            self.optimizers.add(
                id="grade",
                optimizer=optim.Adam(
                    self.networks.get_parameters("grade"),
                    lr=cfg["optimizer"]["lr"],
                    betas=(cfg["optimizer"]["beta1"], cfg["optimizer"]["beta2"]),
                    weight_decay=cfg["optimizer"]["weight_decay"],
                ),
                additional_info=cfg["optimizer"],
            )
            self.grade_criterion = self.select_criterion(cfg["criterion"])
            if cfg["scheduler"]["enabled"]:
                self.schedulers.add(
                    id="grade",
                    scheduler=optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer=self.optimizers.get("grade"),
                        mode=cfg["scheduler"]["mode"],
                        factor=cfg["scheduler"]["factor"],
                        patience=cfg["scheduler"]["patience"],
                        verbose=True,
                    ),
                    metric_id=cfg["scheduler"]["metric_name"],
                    metric_category=cfg["scheduler"]["metric_category"],
                    metric_stage=cfg["scheduler"]["metric_stage"],
                    additional_info=cfg["scheduler"],
                )

    def training_step(self, progress_bar: PBarType, batch: Any) -> Dict[str, Any]:
        results = {}

        t_m_brain = batch["mask_brain_shape"].to(self.device)
        t_m_shape = batch["mask_tumor_shape"].to(self.device)
        t_m_grade = batch["mask_tumor_grade"].to(self.device)
        t_m_circles = batch["mask_circles"].to(self.device)

        t_m_ed = batch["mask_ed"].to(self.device)
        t_m_et = batch["mask_et"].to(self.device)
        t_m_net = batch["mask_net"].to(self.device)

        t_m_background = 1 - t_m_shape

        if self.shape_enabled:
            shape_seg = self.networks.get("shape")
            shape_seg.train()
            shape_opt = self.optimizers.get("shape")
            shape_opt.zero_grad()

            p_m_shape = shape_seg(torch.cat((t_m_brain, t_m_circles), 1))
            t_m_shape_all = torch.cat((t_m_background, t_m_shape), 1)

            shape_loss = self.shape_criterion(p_m_shape, t_m_shape_all)
            shape_loss.backward()
            shape_opt.step()

            results["shape_loss"] = shape_loss.item()

            shape_dice_score = calc_dice_score(
                p_m_shape.detach(), t_m_shape_all, "mean"
            )
            results["shape_dice_score"] = shape_dice_score.mean().item()

            cm = self.form_cm(p_m_shape, t_m_shape, 2)
            results["shape_confusion_matrix"] = cm
            results["shape_precision"] = np.mean(calc_precision(cm, "mean"))
            results["shape_recall"] = np.mean(calc_recall(cm, "mean"))

            if self.on_last_batch:
                self.shape_img2tb(p_m_shape, t_m_shape)

        if self.grade_enabled:
            grade_seg = self.networks.get("grade")
            grade_seg.train()
            grade_opt = self.optimizers.get("grade")
            grade_opt.zero_grad()

            p_m_grade = grade_seg(torch.cat((t_m_shape, t_m_circles), 1))
            t_m_grade_all = torch.cat((t_m_background, t_m_ed, t_m_et, t_m_net), 1)
            grade_loss = self.grade_criterion(p_m_grade, t_m_grade_all)

            grade_loss.backward()
            grade_opt.step()

            results["grade_loss"] = grade_loss.item()

            grade_dice_score = calc_dice_score(
                p_m_grade.detach(), t_m_grade_all, "mean"
            )
            results["grade_dice_score_ed"] = grade_dice_score[1].item()
            results["grade_dice_score_et"] = grade_dice_score[2].item()
            results["grade_dice_score_net"] = grade_dice_score[3].item()

            cm = self.form_cm(p_m_grade, t_m_grade, 4)
            results["grade_confusion_matrix"] = cm

            precision = calc_precision(cm, "mean")
            recall = calc_recall(cm, "mean")
            results["grade_precision_ed"] = precision[1]
            results["grade_recall_ed"] = recall[1]
            results["grade_precision_et"] = precision[2]
            results["grade_recall_et"] = recall[2]
            results["grade_precision_net"] = precision[3]
            results["grade_recall_net"] = recall[3]

            if self.on_last_batch:
                self.grade_img2tb(p_m_grade, t_m_grade)

        return results

    def validation_step(self, progress_bar: PBarType, batch: Any) -> Dict[str, Any]:
        results = {}

        t_m_brain = batch["mask_brain_shape"].to(self.device)
        t_m_shape = batch["mask_tumor_shape"].to(self.device)
        t_m_grade = batch["mask_tumor_grade"].to(self.device)
        t_m_circles = batch["mask_circles"].to(self.device)

        t_m_ed = batch["mask_ed"].to(self.device)
        t_m_et = batch["mask_et"].to(self.device)
        t_m_net = batch["mask_net"].to(self.device)

        t_m_background = 1 - t_m_shape

        if self.shape_enabled:
            shape_seg = self.networks.get("shape")
            shape_seg.eval()

            p_m_shape = shape_seg(torch.cat((t_m_brain, t_m_circles), 1))
            t_m_shape_all = torch.cat((t_m_background, t_m_shape), 1)

            shape_loss = self.shape_criterion(p_m_shape, t_m_shape_all)
            results["shape_loss"] = shape_loss.item()

            shape_dice_score = calc_dice_score(
                p_m_shape.detach(), t_m_shape_all, "mean"
            )
            results["shape_dice_score"] = shape_dice_score.mean().item()

            cm = self.form_cm(p_m_shape.detach(), t_m_shape, 2)
            results["shape_confusion_matrix"] = cm
            results["shape_precision"] = np.mean(calc_precision(cm, "mean"))
            results["shape_recall"] = np.mean(calc_recall(cm, "mean"))

            if self.on_last_batch:
                self.shape_img2tb(p_m_shape, t_m_shape)

        if self.grade_enabled:
            grade_seg = self.networks.get("grade")
            grade_seg.eval()

            p_m_grade = grade_seg(torch.cat((t_m_shape, t_m_circles), 1))
            t_m_grade_all = torch.cat((t_m_background, t_m_ed, t_m_et, t_m_net), 1)

            grade_loss = self.grade_criterion(p_m_grade, t_m_grade_all)
            results["grade_loss"] = grade_loss.item()

            grade_dice_score = calc_dice_score(
                p_m_grade.detach(), t_m_grade_all, "mean"
            )
            results["grade_dice_score_ed"] = grade_dice_score[1].item()
            results["grade_dice_score_et"] = grade_dice_score[2].item()
            results["grade_dice_score_net"] = grade_dice_score[3].item()

            cm = self.form_cm(p_m_grade, t_m_grade, 4)
            results["grade_confusion_matrix"] = cm

            precision = calc_precision(cm, "mean")
            recall = calc_recall(cm, "mean")
            results["grade_precision_ed"] = precision[1]
            results["grade_recall_ed"] = recall[1]
            results["grade_precision_et"] = precision[2]
            results["grade_recall_et"] = recall[2]
            results["grade_precision_net"] = precision[3]
            results["grade_recall_net"] = recall[3]

            if self.on_last_batch:
                self.grade_img2tb(p_m_grade, t_m_grade)

        return results

    def on_training_epoch_end(self):
        if (
            self.shape_enabled
            and self.tb_writer is not None
            # and (self.on_saving_epoch or self.on_last_epoch)
        ):
            self.tb_writer.add_figure(
                "train/shape_confusion_matrix",
                create_confusion_matrix_image(
                    self.metrics.get("shape_confusion_matrix", "epoch", "train"),
                    ["background", "tumor"],
                ),
                self.current_epoch,
            )
        if (
            self.grade_enabled
            and self.tb_writer is not None
            # and (self.on_saving_epoch or self.on_last_epoch)
        ):
            self.tb_writer.add_figure(
                "train/grade_confusion_matrix",
                create_confusion_matrix_image(
                    self.metrics.get("grade_confusion_matrix", "epoch", "train"),
                    ["background", "ED", "ET", "NET"],
                ),
                self.current_epoch,
            )

    def on_validation_epoch_end(self):
        if (
            self.shape_enabled
            and self.tb_writer is not None
            # and (self.on_saving_epoch or self.on_last_epoch)
        ):
            self.tb_writer.add_figure(
                "val/shape_confusion_matrix",
                create_confusion_matrix_image(
                    self.metrics.get("shape_confusion_matrix", "epoch", "val"),
                    ["background", "tumor"],
                ),
                self.current_epoch,
            )
        if (
            self.grade_enabled
            and self.tb_writer is not None
            # and (self.on_saving_epoch or self.on_last_epoch)
        ):
            self.tb_writer.add_figure(
                "val/grade_confusion_matrix",
                create_confusion_matrix_image(
                    self.metrics.get("grade_confusion_matrix", "epoch", "val"),
                    ["background", "ED", "ET", "NET"],
                ),
                self.current_epoch,
            )

    @staticmethod
    def form_cm(p: torch.Tensor, t: torch.Tensor, n_class: int):
        return calc_confusion_matrix(
            t.detach().cpu(), p.detach().cpu().argmax(1, True), n_class, False, "mean"
        )

    def shape_img2tb(self, p: torch.Tensor, t: torch.Tensor):
        if self.tb_writer is not None:
            self.tb_writer.add_image(
                f"{self._stage}/shape_masks",
                create_grid_shape(
                    p.detach().cpu().argmax(1, True),
                    t.detach().cpu(),
                    min(p.shape[0], 6),
                ),
                self.current_epoch,
            )

    def grade_img2tb(self, p: torch.Tensor, t: torch.Tensor):
        if self.tb_writer is not None:
            self.tb_writer.add_image(
                f"{self._stage}/grade_masks",
                create_grid_grade(
                    p.detach().cpu(), t.detach().cpu(), min(p.shape[0], 6),
                ),
                self.current_epoch,
            )

    @staticmethod
    def select_segmenter(cfg: dict, in_channels: int, classes: int):
        r"""
        TODO documentation
        """
        params = {
            "encoder_name": cfg["encoder_name"],
            "encoder_depth": cfg["encoder_depth"],
            "encoder_weights": None,
            "decoder_channels": cfg["decoder_channels"],
            "decoder_use_batchnorm": cfg["decoder_use_batchnorm"],
            "in_channels": in_channels,
            "classes": classes,
        }
        if cfg["network_type"] == "Unet":
            return smp.Unet(**params)
        elif cfg["network_type"] == "Unet++":
            return smp.UnetPlusPlus(**params)
        else:
            raise ValueError(
                f"Unknown network type '{cfg['network_type']}' for segmenter"
            )

    def select_criterion(self, cfg: dict):
        r"""
        TODO documentation
        """
        loss_names = cfg["loss_names"]
        loss_weights = cfg["loss_weights"]
        if loss_names is not None:
            loss_names = list(loss_names)

        losses = []

        for name in loss_names:
            if name == "cross_entropy":
                assert "cross_entropy" in cfg
                weights = cfg["cross_entropy"]["weights"]
                if weights is not None:
                    weights = torch.FloatTensor(weights).to(self.device)
                reduction = cfg["cross_entropy"]["reduction"]
                losses.append(nn.CrossEntropyLoss(weight=weights, reduction=reduction))
            elif name == "dice":
                assert "dice" in cfg
                weights = cfg["dice"]["weights"]
                if weights is not None:
                    weights = torch.FloatTensor(weights).to(self.device)
                ignore_index = cfg["dice"]["ignore_index"]
                apply_softmax = cfg["dice"]["apply_softmax"]
                reduction = cfg["dice"]["reduction"]
                losses.append(
                    DiceLoss(
                        weights=weights,
                        ignore_index=ignore_index,
                        reduction=reduction,
                        apply_softmax=apply_softmax,
                    )
                )
            elif name == "focal":
                assert "focal" in cfg
                from kornia.losses import FocalLoss

                alpha = cfg["focal"]["alpha"]
                gamma = cfg["focal"]["gamma"]
                reduction = cfg["focal"]["reduction"]
                losses.append(
                    TargetArgmaxLossWrapper(
                        FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
                    )
                )
            elif name == "hausdorff":
                from kornia.losses import HausdorffERLoss

                alpha = cfg["hausdorff"]["alpha"]
                k = cfg["hausdorff"]["k"]
                reduction = cfg["hausdorff"]["reduction"]
                losses.append(
                    TargetArgmaxLossWrapper(
                        HausdorffERLoss(alpha=alpha, k=k, reduction=reduction), True
                    )
                )

        if len(losses) == 1:
            return losses[0]
        else:
            assert len(losses) == len(loss_weights)
            return JointLoss(losses=losses, weights=loss_weights)
