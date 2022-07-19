from typing import Any, Dict, Optional
import numpy as np
from torch import optim

import torch
from torch import nn
from torch.utils.data import DataLoader
from focal_frequency_loss import FocalFrequencyLoss as FFL

from trainer import BaseTrainer, TimerCallback
from trainer.typing import TBLevelType, PathType, PBarType

from src.models.baseline import (
    BaselineInpaintingDiscriminator,
    BaselineInpaintingGenerator,
    BaselineVGG16,
    BaselineGradeSegmenter,
    BaselineShapeSegmenter,
)
from src.models.losses import (
    BinaryDiceLoss,
    calc_confusion_matrix,
    calc_precision,
    calc_recall,
)
from src.utils.visuals import (
    create_confusion_matrix_image,
    create_grid_grade,
    create_grid_inpainting,
    create_grid_shape,
)


class BaselineExperiment(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.callbacks.append(TimerCallback())

        assert self.params is not None

        self.shape_enabled = self.params["shape_segmenter"]["enabled"]
        self.grade_enabled = self.params["grade_segmenter"]["enabled"]
        self.inpainting_enabled = self.params["inpainting"]["enabled"]

        assert self.shape_enabled or self.grade_enabled or self.inpainting_enabled

        self.ffl = None
        if "ffl" in self.params and self.params["ffl"]["enabled"]:
            self.ffl_w = self.params["ffl"]["weight"]
            self.ffl = FFL(loss_weight=1.0, alpha=1.0)  # initialize nn.Module class

        if self.shape_enabled:
            cfg = self.params["shape_segmenter"]
            self.networks.add(
                id="shape",
                network=BaselineShapeSegmenter(6).to(self.device),
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
                    metric_stage=cfg["scheduler"]["stage"],
                    additional_info=cfg["scheduler"],
                )

        if self.grade_enabled:
            cfg = self.params["grade_segmenter"]
            self.networks.add(
                id="grade",
                network=BaselineGradeSegmenter().to(self.device),
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

        if self.inpainting_enabled:
            cfg_gen = self.params["inpainting"]["generator"]

            self.networks.add(
                id="inpainting_gen",
                network=BaselineInpaintingGenerator().to(self.device),
                additional_info=cfg_gen,
            )

            self.optimizers.add(
                id="inpainting_gen",
                optimizer=optim.Adam(
                    self.networks.get_parameters("inpainting_gen"),
                    lr=cfg_gen["optimizer"]["lr"],
                    betas=(
                        cfg_gen["optimizer"]["beta1"],
                        cfg_gen["optimizer"]["beta2"],
                    ),
                    weight_decay=cfg_gen["optimizer"]["weight_decay"],
                ),
                additional_info=cfg_gen["optimizer"],
            )
            if cfg_gen["scheduler"]["enabled"]:
                self.schedulers.add(
                    id="inpainting_gen",
                    scheduler=optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer=self.optimizers.get("inpainting_gen"),
                        mode=cfg_gen["scheduler"]["mode"],
                        factor=cfg_gen["scheduler"]["factor"],
                        patience=cfg_gen["scheduler"]["patience"],
                        verbose=True,
                    ),
                    metric_id=cfg_gen["scheduler"]["metric_name"],
                    metric_category=cfg_gen["scheduler"]["metric_category"],
                    metric_stage=cfg_gen["scheduler"]["metric_stage"],
                    additional_info=cfg_gen["scheduler"],
                )

            cfg_disc = self.params["inpainting"]["discriminator"]
            self.networks.add(
                id="inpainting_disc",
                network=BaselineInpaintingDiscriminator().to(self.device),
                additional_info=cfg_disc,
            )
            self.optimizers.add(
                id="inpainting_disc",
                optimizer=optim.Adam(
                    self.networks.get_parameters("inpainting_disc"),
                    lr=cfg_disc["optimizer"]["lr"],
                    betas=(
                        cfg_disc["optimizer"]["beta1"],
                        cfg_disc["optimizer"]["beta2"],
                    ),
                    weight_decay=cfg_disc["optimizer"]["weight_decay"],
                ),
                additional_info=cfg_disc["optimizer"],
            )

            if cfg_disc["scheduler"]["enabled"]:
                self.schedulers.add(
                    id="inpainting_disc",
                    scheduler=optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer=self.optimizers.get("inpainting_disc"),
                        mode=cfg_disc["scheduler"]["mode"],
                        factor=cfg_disc["scheduler"]["factor"],
                        patience=cfg_disc["scheduler"]["patience"],
                        verbose=True,
                    ),
                    metric_id=cfg_disc["scheduler"]["metric_name"],
                    metric_category=cfg_disc["scheduler"]["metric_category"],
                    metric_stage=cfg_disc["scheduler"]["metric_stage"],
                    additional_info=cfg_disc["scheduler"],
                )

            self.networks.add(
                id="vgg16",
                network=BaselineVGG16(requires_grad=False).to(self.device),
                allow_save=False,
                allow_load=False,
                allow_stats=False,
            )

        self.binary_dice_loss = BinaryDiceLoss()
        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = nn.L1Loss(reduction="sum")

    def training_step(self, progress_bar: PBarType, batch: Any) -> Dict[str, Any]:
        results = {}

        t_t1 = batch["t1"].to(self.device)
        t_t1ce = batch["t1ce"].to(self.device)
        t_t2 = batch["t2"].to(self.device)
        t_flair = batch["flair"].to(self.device)

        t_m_brain = batch["mask_brain_shape"].to(self.device)
        t_m_shape = batch["mask_tumor_shape"].to(self.device)
        t_m_grade = batch["mask_tumor_grade"].to(self.device)
        t_m_grade_mapped = self.map_grade(batch["mask_tumor_grade"]).to(self.device)
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
              
            p_m_shape = shape_seg(torch.cat((t_m_brain, t_m_circles, t_t1, t_t1ce, t_t2, t_flair), 1)) ## adding original images too
            #p_m_shape = shape_seg(torch.cat((t_m_brain, t_m_circles), 1)) # papers code
            shape_loss = self.l1_loss(p_m_shape, t_m_shape)

            shape_loss.backward()
            shape_opt.step()
            results["shape_loss"] = shape_loss.item()

            shape_dice_score = 1 - self.shape_dice(p_m_shape, t_m_shape)
            results["shape_dice_score"] = shape_dice_score.item()
            cm = self.shape_cm(p_m_shape.detach().cpu(), t_m_shape.detach().cpu())
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
            grade_loss = self.l1_loss(p_m_grade, t_m_grade_mapped)

            grade_loss.backward()
            grade_opt.step()

            results["grade_loss"] = grade_loss.item()

            grade_dice_score = self.grade_dice(p_m_grade, t_m_net, t_m_et, t_m_ed)
            results["grade_dice_score_ed"] = grade_dice_score[0].item()
            results["grade_dice_score_et"] = grade_dice_score[1].item()
            results["grade_dice_score_net"] = grade_dice_score[2].item()

            cm = self.grade_cm(p_m_grade.cpu(), t_m_grade.cpu())
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
                self.grade_img2tb(p_m_grade, t_m_grade_mapped)

        if self.inpainting_enabled:
            inp_gen = self.networks.get("inpainting_gen")
            inp_gen.train()
            inp_gen_opt = self.optimizers.get("inpainting_gen")
            inp_gen_opt.zero_grad()

            inp_disc = self.networks.get("inpainting_disc")
            inp_disc.train()
            inp_disc_opt = self.optimizers.get("inpainting_disc")
            inp_disc_opt.zero_grad()

            vgg16 = self.networks.get("vgg16")
            vgg16.eval()

            t_brain = torch.cat((t_t1, t_t1ce, t_t2, t_flair), 1)
            brain_w_blank_mask = t_brain * t_m_background

            p_brain = inp_gen(brain_w_blank_mask, t_m_grade_mapped)
            inp_disc_real_loss = self.realTargetLoss(
                inp_disc(t_brain, t_m_grade_mapped)
            )
            inp_disc_fake_loss = self.fakeTargetLoss(
                inp_disc(p_brain.detach(), t_m_grade_mapped)
            )
            inp_disc_loss = inp_disc_real_loss + inp_disc_fake_loss

            inp_disc_loss.backward()
            inp_disc_opt.step()

            inp_gb_loss = self.l1_loss(p_brain, t_brain) / torch.sum(t_m_brain)
            inp_lc_loss = self.l1_loss(
                p_brain * t_m_shape, t_brain * t_m_shape
            ) / torch.sum(t_m_shape)
            inp_adv_loss = self.realTargetLoss(inp_disc(p_brain, t_m_grade_mapped))

            inp_content_loss = self.calc_content_loss(
                vgg16, p_brain, t_brain, t_m_shape, t_m_brain
            )

            inp_gen_loss = inp_gb_loss + inp_lc_loss + inp_adv_loss + inp_content_loss
            if self.ffl is not None:
                ffl_loss = self.ffl(p_brain, t_brain)
                results["inp_ffl_loss"] = self.ffl_w * ffl_loss.item()
                inp_gen_loss += self.ffl_w * ffl_loss

            inp_gen_loss.backward()
            inp_gen_opt.step()

            results["inp_disc_fake_loss"] = inp_disc_fake_loss.item()
            results["inp_disc_real_loss"] = inp_disc_real_loss.item()
            results["inp_disc_loss"] = inp_disc_loss.item()
            results["inp_gb_loss"] = inp_gb_loss.item()
            results["inp_lc_loss"] = inp_lc_loss.item()
            results["inp_adv_loss"] = inp_adv_loss.item()
            results["inp_content_loss"] = inp_content_loss.item()
            results["inp_gen_loss"] = inp_gen_loss.item()

            if self.on_last_batch:
                self.inpainting_img2tb(
                    p_brain * t_m_shape + t_brain * t_m_background, t_brain
                )

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
        t_m_grade_mapped = self.map_grade(batch["mask_tumor_grade"]).to(self.device)
        t_m_circles = batch["mask_circles"].to(self.device)

        t_m_ed = batch["mask_ed"].to(self.device)
        t_m_et = batch["mask_et"].to(self.device)
        t_m_net = batch["mask_net"].to(self.device)

        t_m_background = 1 - t_m_shape

        if self.shape_enabled:
            shape_seg = self.networks.get("shape")
            shape_seg.eval()

            #p_m_shape = shape_seg(torch.cat((t_m_brain, t_m_circles), 1))
            p_m_shape = shape_seg(torch.cat((t_m_brain, t_m_circles, t_t1, t_t1ce, t_t2, t_flair), 1))
            shape_loss = self.l1_loss(p_m_shape, t_m_shape)
            results["shape_loss"] = shape_loss.item()

            shape_dice_score = 1 - self.shape_dice(p_m_shape, t_m_shape)
            cm = self.shape_cm(p_m_shape.cpu(), t_m_shape.cpu())

            results["shape_confusion_matrix"] = cm
            results["shape_precision"] = np.mean(calc_precision(cm, "mean"))
            results["shape_recall"] = np.mean(calc_recall(cm, "mean"))
            results["shape_dice_score"] = shape_dice_score.item()

            if self.on_last_batch:
                self.shape_img2tb(p_m_shape, t_m_shape)

        if self.grade_enabled:
            grade_seg = self.networks.get("grade")
            grade_seg.eval()

            p_m_grade = grade_seg(torch.cat((t_m_shape, t_m_circles), 1))
            grade_loss = self.l1_loss(p_m_grade, t_m_grade_mapped)
            results["grade_loss"] = grade_loss.item()

            grade_dice_score = self.grade_dice(p_m_grade, t_m_net, t_m_et, t_m_ed)
            results["grade_dice_score_ed"] = grade_dice_score[0].item()
            results["grade_dice_score_et"] = grade_dice_score[1].item()
            results["grade_dice_score_net"] = grade_dice_score[2].item()

            cm = self.grade_cm(p_m_grade.cpu(), t_m_grade.cpu())
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
                self.grade_img2tb(p_m_grade, t_m_grade_mapped)

        if self.inpainting_enabled:
            inp_gen = self.networks.get("inpainting_gen")
            inp_gen.eval()
            inp_disc = self.networks.get("inpainting_disc")
            inp_disc.eval()
            vgg16 = self.networks.get("vgg16")
            vgg16.eval()

            t_brain = torch.cat((t_t1, t_t1ce, t_t2, t_flair), 1)
            brain_w_blank_mask = t_brain * t_m_background

            p_brain = inp_gen(brain_w_blank_mask, t_m_grade_mapped)
            inp_disc_real_loss = self.realTargetLoss(
                inp_disc(t_brain, t_m_grade_mapped)
            )
            inp_disc_fake_loss = self.fakeTargetLoss(
                inp_disc(p_brain.detach(), t_m_grade_mapped)
            )
            inp_disc_loss = inp_disc_real_loss + inp_disc_fake_loss

            inp_gb_loss = self.l1_loss(p_brain, t_brain) / torch.sum(t_m_brain)
            inp_lc_loss = self.l1_loss(
                p_brain * t_m_shape, t_brain * t_m_shape
            ) / torch.sum(t_m_shape)
            inp_adv_loss = self.realTargetLoss(inp_disc(p_brain, t_m_grade_mapped))

            inp_content_loss = self.calc_content_loss(
                vgg16, p_brain, t_brain, t_m_shape, t_m_brain
            )

            inp_gen_loss = inp_gb_loss + inp_lc_loss + inp_adv_loss + inp_content_loss

            results["inp_disc_fake_loss"] = inp_disc_fake_loss.item()
            results["inp_disc_real_loss"] = inp_disc_real_loss.item()
            results["inp_disc_loss"] = inp_disc_loss.item()
            results["inp_gb_loss"] = inp_gb_loss.item()
            results["inp_lc_loss"] = inp_lc_loss.item()
            results["inp_adv_loss"] = inp_adv_loss.item()
            results["inp_content_loss"] = inp_content_loss.item()
            results["inp_gen_loss"] = inp_gen_loss.item()

            if self.on_last_batch:
                self.inpainting_img2tb(
                    p_brain * t_m_shape + t_brain * t_m_background, t_brain
                )

        return results

    def on_training_epoch_end(self):
        if self.shape_enabled and self.tb_writer is not None:
            self.tb_writer.add_figure(
                "train/shape_confusion_matrix",
                create_confusion_matrix_image(
                    self.metrics.get("shape_confusion_matrix", "epoch", "train"),
                    ["background", "tumor"],
                ),
                self.current_epoch,
            )
        if self.grade_enabled and self.tb_writer is not None:
            self.tb_writer.add_figure(
                "train/grade_confusion_matrix",
                create_confusion_matrix_image(
                    self.metrics.get("grade_confusion_matrix", "epoch", "train"),
                    ["background", "ED", "ET", "NET"],
                ),
                self.current_epoch,
            )

    def on_validation_epoch_end(self):
        if self.shape_enabled and self.tb_writer is not None:
            self.tb_writer.add_figure(
                "val/shape_confusion_matrix",
                create_confusion_matrix_image(
                    self.metrics.get("shape_confusion_matrix", "epoch", "val"),
                    ["background", "tumor"],
                ),
                self.current_epoch,
            )
        if self.grade_enabled and self.tb_writer is not None:
            self.tb_writer.add_figure(
                "val/grade_confusion_matrix",
                create_confusion_matrix_image(
                    self.metrics.get("grade_confusion_matrix", "epoch", "val"),
                    ["background", "ED", "ET", "NET"],
                ),
                self.current_epoch,
            )

    @staticmethod
    def one2three(x) -> torch.Tensor:
        temp = torch.cat([x, x, x], dim=1).to(x.device)
        return temp

    def shape_dice(self, p: torch.Tensor, t: torch.Tensor):
        """
        non-differentiable
        """
        return self.binary_dice_loss(torch.where(p > 0.3, 1.0, 0.0), t)

    def grade_dice(
        self, p: torch.Tensor, net: torch.Tensor, et: torch.Tensor, ed: torch.Tensor
    ):
        """
        non-differentiable
        """
        quantized = self.quantize_grade(p, True)
        return (
            1 - self.binary_dice_loss(torch.where(quantized == 0.5, 1.0, 0.0), ed),
            1 - self.binary_dice_loss(torch.where(quantized == 0.75, 1.0, 0.0), et),
            1 - self.binary_dice_loss(torch.where(quantized == 1.0, 1.0, 0.0), net),
        )

    @staticmethod
    def shape_cm(p: torch.Tensor, t: torch.Tensor):
        mapped = torch.zeros(p.shape, dtype=torch.long)
        mapped[(p > 0.5).nonzero(as_tuple=True)] = 1
        return calc_confusion_matrix(t, mapped, 2, False, "mean")

    @staticmethod
    def grade_cm(p: torch.Tensor, t: torch.Tensor):
        mapped = torch.zeros(p.shape, dtype=torch.long)
        mapped[(p > 0.36).nonzero(as_tuple=True)] = 1
        mapped[(p > 0.63).nonzero(as_tuple=True)] = 2
        mapped[(p > 0.85).nonzero(as_tuple=True)] = 3
        return calc_confusion_matrix(t, mapped, 4, False, "mean")

    @staticmethod
    def quantize_grade(x: torch.Tensor, requires_grad: bool = False):
        out = torch.zeros(x.shape, dtype=torch.float, requires_grad=requires_grad).to(
            x.device
        )
        out[(x > 0.36).nonzero(as_tuple=True)] = 0.5
        out[(x > 0.63).nonzero(as_tuple=True)] = 0.75
        out[(x > 0.85).nonzero(as_tuple=True)] = 1.0
        return out

    @staticmethod
    def map_grade(x: torch.Tensor, requires_grad: bool = False):
        out = torch.zeros(x.shape, dtype=torch.float, requires_grad=requires_grad).to(
            x.device
        )
        out[(x == 1).nonzero(as_tuple=True)] = 0.5
        out[(x == 2).nonzero(as_tuple=True)] = 0.75
        out[(x == 3).nonzero(as_tuple=True)] = 1.0
        return out

    def realTargetLoss(self, x):
        device = x.device
        target = torch.Tensor(x.shape[0], 1).fill_(1.0).to(device)
        return self.mse_loss(x, target)

    def fakeTargetLoss(self, x):
        device = x.device
        target = torch.Tensor(x.shape[0], 1).fill_(0.0).to(device)
        return self.mse_loss(x, target)

    def calc_content_loss(self, vgg, p_brain, t_brain, t_m_shape, t_m_brain):
        t_m_t1, t_m_t1ce, t_m_t2, t_m_flair = torch.split(
            t_brain * t_m_shape, split_size_or_sections=1, dim=1
        )
        p_m_t1, p_m_t1ce, p_m_t2, p_m_flair = torch.split(
            p_brain * t_m_shape, split_size_or_sections=1, dim=1
        )

        (
            t_t1_ft,
            p_t1_ft,
            t_t1ce_ft,
            p_t1ce_ft,
            t_t2_ft,
            p_t2_ft,
            t_flair_ft,
            p_flair_ft,
        ) = (
            vgg(self.one2three(t_m_t1).to(self.device)),
            vgg(self.one2three(p_m_t1).to(self.device)),
            vgg(self.one2three(t_m_t1ce).to(self.device)),
            vgg(self.one2three(p_m_t1ce).to(self.device)),
            vgg(self.one2three(t_m_t2).to(self.device)),
            vgg(self.one2three(p_m_t2).to(self.device)),
            vgg(self.one2three(t_m_flair).to(self.device)),
            vgg(self.one2three(p_m_flair).to(self.device)),
        )
        return (
            self.l1_loss(t_t1_ft.relu2_2, p_t1_ft.relu2_2)
            + self.l1_loss(t_t1ce_ft.relu2_2, p_t1ce_ft.relu2_2)
            + self.l1_loss(t_t2_ft.relu2_2, p_t2_ft.relu2_2)
            + self.l1_loss(t_flair_ft.relu2_2, p_flair_ft.relu2_2)
        ) / torch.sum(t_m_brain)

    def shape_img2tb(self, p: torch.Tensor, t: torch.Tensor) -> None:
        if self.tb_writer is not None:
            self.tb_writer.add_image(
                f"{self._stage}/shape_masks",
                create_grid_shape(
                    p.detach().cpu(), t.detach().cpu(), min(p.shape[0], 6), 1, 0.3,
                ),
                self.current_epoch,
            )

    def grade_img2tb(self, p: torch.Tensor, t: torch.Tensor) -> None:
        if self.tb_writer is not None:
            self.tb_writer.add_image(
                f"{self._stage}/grade_masks",
                create_grid_grade(
                    self.quantize_grade(p.detach().cpu()),
                    t.detach().cpu(),
                    min(p.shape[0], 6),
                ),
                self.current_epoch,
            )

    def inpainting_img2tb(self, p: torch.Tensor, t: torch.Tensor) -> None:
        if self.tb_writer is not None:
            self.tb_writer.add_image(
                f"{self._stage}/inpainting",
                create_grid_inpainting(
                    p.detach().cpu(), t.detach().cpu(), min(p.shape[0], 6)
                ),
                self.current_epoch,
            )
