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

from src.utils.visuals import create_grid_segmentation
from src.utils.losses import (
    JointLoss,
    DiceLoss,
)
from src.utils.metrics import (
    calc_confusion_matrix,
    calc_dice_score,
    calc_precision,
    calc_recall,
)


class SegmentationExperiment(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.callbacks.append(TimerCallback())

        assert self.params is not None

        cfg = self.params["segmenter"]

        self.t1_enabled = self.params["t1_enabled"]
        self.t1ce_enabled = self.params["t1ce_enabled"]
        self.t2_enabled = self.params["t2_enabled"]
        self.flair_enabled = self.params["flair_enabled"]

        assert (
            self.t1_enabled
            or self.t1ce_enabled
            or self.t2_enabled
            or self.flair_enabled
        )

        # T1 TUMOR
        if self.t1_enabled:
            self.networks.add(
                id="seg_t1",
                network=self.select_segmenter(cfg, 1, 4).to(self.device),
                additional_info=cfg,
            )
            self.optimizers.add(
                id="seg_t1",
                optimizer=optim.Adam(
                    self.networks.get_parameters("seg_t1"),
                    lr=cfg["optimizer"]["lr"],
                    betas=(cfg["optimizer"]["beta1"], cfg["optimizer"]["beta2"]),
                    weight_decay=cfg["optimizer"]["weight_decay"],
                ),
                additional_info=cfg["optimizer"],
            )
            self.seg_t1_criterion = self.select_criterion(cfg["criterion"])

        # T1CE TUMOR
        if self.t1ce_enabled:
            self.networks.add(
                id="seg_t1ce",
                network=self.select_segmenter(cfg, 1, 4).to(self.device),
                additional_info=cfg,
            )
            self.optimizers.add(
                id="seg_t1ce",
                optimizer=optim.Adam(
                    self.networks.get_parameters("seg_t1ce"),
                    lr=cfg["optimizer"]["lr"],
                    betas=(cfg["optimizer"]["beta1"], cfg["optimizer"]["beta2"]),
                    weight_decay=cfg["optimizer"]["weight_decay"],
                ),
                additional_info=cfg["optimizer"],
            )
            self.seg_t1ce_criterion = self.select_criterion(cfg["criterion"])

        # T2 TUMOR
        if self.t2_enabled:
            self.networks.add(
                id="seg_t2",
                network=self.select_segmenter(cfg, 1, 4).to(self.device),
                additional_info=cfg,
            )
            self.optimizers.add(
                id="seg_t2",
                optimizer=optim.Adam(
                    self.networks.get_parameters("seg_t2"),
                    lr=cfg["optimizer"]["lr"],
                    betas=(cfg["optimizer"]["beta1"], cfg["optimizer"]["beta2"]),
                    weight_decay=cfg["optimizer"]["weight_decay"],
                ),
                additional_info=cfg["optimizer"],
            )
            self.seg_t2_criterion = self.select_criterion(cfg["criterion"])

        # FLAIR TUMOR
        if self.flair_enabled:
            self.networks.add(
                id="seg_flair",
                network=self.select_segmenter(cfg, 1, 4).to(self.device),
                additional_info=cfg,
            )
            self.optimizers.add(
                id="seg_flair",
                optimizer=optim.Adam(
                    self.networks.get_parameters("seg_flair"),
                    lr=cfg["optimizer"]["lr"],
                    betas=(cfg["optimizer"]["beta1"], cfg["optimizer"]["beta2"]),
                    weight_decay=cfg["optimizer"]["weight_decay"],
                ),
                additional_info=cfg["optimizer"],
            )
            self.seg_flair_criterion = self.select_criterion(cfg["criterion"])

    def training_step(self, progress_bar: PBarType, batch: Any) -> Dict[str, Any]:
        results = {}

        t_t1 = batch["t1"].to(self.device)
        t_t1ce = batch["t1ce"].to(self.device)
        t_t2 = batch["t2"].to(self.device)
        t_flair = batch["flair"].to(self.device)

        t_m_brain = batch["mask_brain_shape"].to(self.device)
        t_m_shape = batch["mask_tumor_shape"].to(self.device)
        t_m_grade = batch["mask_tumor_grade"].to(self.device)

        t_m_ed = batch["mask_ed"].to(self.device)
        t_m_et = batch["mask_et"].to(self.device)
        t_m_net = batch["mask_net"].to(self.device)

        t_m_background = 1 - t_m_shape
        t_m_grade_all = torch.cat((t_m_background, t_m_ed, t_m_et, t_m_net), 1)

        if self.t1_enabled:
            seg_t1 = self.networks.get("seg_t1")
            seg_t1.train()
            t1_opt = self.optimizers.get("seg_t1")
            t1_opt.zero_grad()

            p_m_grade = seg_t1(t_t1)

            t1_loss = self.seg_t1_criterion(p_m_grade, t_m_grade_all)
            t1_loss.backward()
            t1_opt.step()

            results["t1_loss"] = t1_loss.item()

            grade_dice_score = calc_dice_score(
                p_m_grade.detach(), t_m_grade_all, "mean"
            )
            results["t1_dice_score_ed"] = grade_dice_score[1].item()
            results["t1_dice_score_et"] = grade_dice_score[2].item()
            results["t1_dice_score_net"] = grade_dice_score[3].item()

            cm = self.form_cm(p_m_grade, t_m_grade, 4)
            results["t1_confusion_matrix"] = cm

            precision = calc_precision(cm, "mean")
            recall = calc_recall(cm, "mean")
            results["t1_precision_ed"] = precision[1]
            results["t1_recall_ed"] = recall[1]
            results["t1_precision_et"] = precision[2]
            results["t1_recall_et"] = recall[2]
            results["t1_precision_net"] = precision[3]
            results["t1_recall_net"] = recall[3]

            if self.on_last_batch:
                self.seg_img2tb("t1", t_t1, p_m_grade, t_m_grade)

        if self.t1ce_enabled:
            seg_t1ce = self.networks.get("seg_t1ce")
            seg_t1ce.train()
            t1ce_opt = self.optimizers.get("seg_t1ce")
            t1ce_opt.zero_grad()

        if self.t2_enabled:
            seg_t2 = self.networks.get("seg_t2")
            seg_t2.train()
            t2_opt = self.optimizers.get("seg_t2")
            t2_opt.zero_grad()

        if self.flair_enabled:
            seg_flair = self.networks.get("seg_flair")
            seg_flair.train()
            flair_opt = self.optimizers.get("seg_flair")
            flair_opt.zero_grad()

        return results

    def validation_step(self, progress_bar: PBarType, batch: Any) -> Dict[str, Any]:
        results = {}

        t_t1 = batch["t1"].to(self.device)
        t_t1ce = batch["t1ce"].to(self.device)
        t_t2 = batch["t2"].to(self.device)
        t_flair = batch["flair"].to(self.device)

        t_m_brain = batch["mask_brain_shape"].to(self.device)
        t_m_shape = batch["mask_tumor_shape"].to(self.device)
        t_m_grade = batch["mask_tumor_grade"].to(self.device)

        t_m_ed = batch["mask_ed"].to(self.device)
        t_m_et = batch["mask_et"].to(self.device)
        t_m_net = batch["mask_net"].to(self.device)

        t_m_background = 1 - t_m_shape

        if self.t1_enabled:
            seg_t1 = self.networks.get("seg_t1")
            seg_t1.eval()

        if self.t1ce_enabled:
            seg_t1ce = self.networks.get("seg_t1ce")
            seg_t1ce.eval()

        if self.t2_enabled:
            seg_t2 = self.networks.get("seg_t2")
            seg_t2.eval()

        if self.flair_enabled:
            seg_flair = self.networks.get("seg_flair")
            seg_flair.eval()

        return results

    @staticmethod
    def form_cm(p: torch.Tensor, t: torch.Tensor, n_class: int):
        return calc_confusion_matrix(
            t.detach().cpu(), p.detach().cpu().argmax(1, True), n_class, False, "mean"
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
        if len(losses) == 1:
            return losses[0]
        else:
            assert len(losses) == len(loss_weights)
            return JointLoss(losses=losses, weights=loss_weights)

    def seg_img2tb(
        self, seg_type: str, sample: torch.Tensor, p: torch.Tensor, t: torch.Tensor
    ):
        if self.tb_writer is not None:
            self.tb_writer.add_image(
                f"{self._stage}/{seg_type}_segmentations",
                create_grid_segmentation(
                    sample.detach().cpu(),
                    p.detach().cpu(),
                    t.detach().cpu(),
                    min(p.shape[0], 6),
                ),
                self.current_epoch,
            )
