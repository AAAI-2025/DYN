# -*- coding: utf-8 -*-
import copy
import warnings
from typing import List

import torch
import torch.nn as nn
import model_adaptation.utils as adaptation_utils
from api import Batch
from loads.define_model import load_pretrained_model
from model_adaptation.base_adaptation import BaseAdaptation
from model_selection.base_selection import BaseSelection
from model_selection.metrics import Metrics
from utils.logging import Logger
from utils.timer import Timer
import torchvision
from einops import rearrange

class NoAdaptation(BaseAdaptation):
    """Standard test-time evaluation (no adaptation)."""

    def __init__(self, meta_conf, model: nn.Module):
        super().__init__(meta_conf, model)
        # self._patch_len = 1
        self._patch_len = self._meta_conf.patch_len if hasattr(self._meta_conf, "patch_len") else 1 
        # self.transform = self.get_aug_transforms(img_shape=self._meta_conf.img_shape)
        self._aug_type = "patch"
    # def convert_iabn(self, module: nn.Module, **kwargs):
    #     """
    #     Recursively convert all BatchNorm to InstanceAwareBatchNorm.
    #     """
    #     module_output = module
    #     if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
    #         IABN = (
    #             adaptation_utils.InstanceAwareBatchNorm2d
    #             if isinstance(module, nn.BatchNorm2d)
    #             else adaptation_utils.InstanceAwareBatchNorm1d
    #         )
    #         module_output = IABN(
    #             num_channels=module.num_features,
    #             k=self._meta_conf.iabn_k,
    #             eps=module.eps,
    #             momentum=module.momentum,
    #             threshold=self._meta_conf.threshold_note,
    #             affine=module.affine,
    #         )

    #         module_output._bn = copy.deepcopy(module)

    #     for name, child in module.named_children():
    #         module_output.add_module(name, self.convert_iabn(child, **kwargs))
    #     del module
    #     return module_output

    def _initialize_model(self, model: nn.Module):
        """Configure model for adaptation."""
        # if hasattr(self._meta_conf, "iabn") and self._meta_conf.iabn:
        #     # check BN layers
        #     bn_flag = False
        #     for name_module, module in model.named_modules():
        #         if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        #             bn_flag = True
        #     if not bn_flag:
        #         warnings.warn(
        #             "IABN needs bn layers, while there is no bn in the base model."
        #         )
        #     self.convert_iabn(model)
        #     load_pretrained_model(self._meta_conf, model)
        model.eval()
        return model.to(self._meta_conf.device)

    def _post_safety_check(self):
        pass

    def initialize(self, seed: int):
        """Initialize the algorithm."""
        self._model = self._initialize_model(model=copy.deepcopy(self._base_model))

    def adapt_and_eval(
        self,
        episodic: bool,
        metrics: Metrics,
        model_selection_method: BaseSelection,
        current_batch: Batch,
        previous_batches: List[Batch],
        logger: Logger,
        timer: Timer,
        domain_indices = None,

    ):
        """The key entry of test-time adaptation."""
        # some simple initialization.
        with timer("test_time_adaptation"):
            with torch.no_grad():
                # self._model.train()
                self._model.eval()
                y_hat = self._model(self.data_aug(current_batch._x))
                # y_hat = self._model(current_batch._x)

        with timer("evaluate_adaptation_result"):
            metrics.eval(current_batch._y, y_hat)
            if self._meta_conf.base_data_name in ["waterbirds"]:
                self.tta_loss_computer.loss(
                    y_hat, current_batch._y, current_batch._g, is_training=False
                )

    @property
    def name(self):
        return "no_adaptation"
    
    def data_aug(self,x):
        x_prime = x.detach()
        if self._aug_type=='occ':
            first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
            final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
            occlusion_window = final_mean.expand(-1, -1, self._occlusion_size, self._occlusion_size)
            x_prime[:, :, self._row_start:self._row_start+self._occlusion_size,self._column_start:self._column_start+self._occlusion_size] = occlusion_window
        
        elif self._aug_type=='patch':
            resize_t = torchvision.transforms.Resize(((x.shape[-1]//self._patch_len)*self._patch_len,(x.shape[-1]//self._patch_len)*self._patch_len))
            resize_o = torchvision.transforms.Resize((x.shape[-1],x.shape[-1]))
            x_prime = resize_t(x_prime)
            x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=self._patch_len, ps2=self._patch_len)
            perm_idx = torch.argsort(torch.rand(x_prime.shape[0],x_prime.shape[1]), dim=-1)
            x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1),perm_idx]
            x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=self._patch_len, ps2=self._patch_len)
            x_prime = resize_o(x_prime)
        
        elif self._aug_type=='pixel':
            x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
            x_prime = x_prime[:,:,torch.randperm(x_prime.shape[-1])]
            x_prime = rearrange(x_prime, 'b c (ps1 ps2) -> b c ps1 ps2', ps1=x.shape[-1], ps2=x.shape[-1])
            
        return x_prime