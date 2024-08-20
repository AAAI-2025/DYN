# -*- coding: utf-8 -*-
import copy
import functools
from typing import List
import os
from ttab.draw.bn_performance import bn_performance
import PIL
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
import ttab.model_adaptation.utils as utils
from api import Batch
from model_adaptation.base_adaptation import BaseAdaptation
from model_selection.base_selection import BaseSelection
from model_selection.metrics import Metrics
from utils.auxiliary import fork_rng_with_seed
from utils.logging import Logger
from utils.timer import Timer
from torchvision.transforms import ColorJitter, Compose, Lambda
from model_adaptation import ClusterAwareBatchNorm
from loads.models.resnet import Bottleneck
from utils.cam import apply_grad_cam
from model_adaptation import utils
from numpy import random
import torchvision.transforms as transforms

# from einops import rearrange
import numpy as np
import torch
import json


class DYN(BaseAdaptation):


    def __init__(self, meta_conf, model: nn.Module):
        self.norm_name = "bn"
        self.layer_control = True
        super(DYN, self).__init__(meta_conf, model)
        self._meta_conf.step = 0
        self.modelBnGroup = [
            (0, 2, "layer1"),  # 3 个 bottleneck
            (3, 6, "layer2"),  # 4 个 bottleneck
            (7, 12, "layer3"),  # 6 个 bottleneck
            (13, 15, "layer4"),  # 3 个 bottleneck
        ]


    def _initialize_model(self, model: nn.Module):
        """Configure model for adaptation."""
        # model.train()
        # disable grad, to (re-)enable only what specified adaptation method updates
        self.convert_ClusterAwareBatchNorm2d(model)

        model.requires_grad_(False)
        # print(model)
        for module in model.modules():
            # if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            #     module.track_running_stats = True
            #     module.weight.requires_grad_(True)
            #     module.bias.requires_grad_(True)
            if isinstance(module, ClusterAwareBatchNorm.ClusterAwareBatchNorm2d):
                module.track_running_stats = False
                module.source_rate.requires_grad_(True)
                # module.test_rate.requires_grad_(True)

        return model.to(self._meta_conf.device)

    def verify_optimizer_params(self, optimizer, adapt_param_names):
        print("[info] Verifying optimizer parameters...")
        for idx, param in enumerate(adapt_param_names):
            try:
                # Accessing parameter by index
                param_tensor = optimizer.param_groups[0]["params"][idx]
                print(f"Parameter {idx} ({param}) indexed successfully={param_tensor}")
            except IndexError:
                print(f"Parameter {idx} ({param}) could not be indexed.")
            except Exception as e:
                print(
                    f"An error occurred while indexing parameter {idx} ({param}): {e}"
                )

    def _initialize_trainable_parameters(self):
        """
        Collect the affine scale + shift parameters from norm layers.

        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        self._adapt_module_names = []
        adapt_params = []
        adapt_param_names = []

        for name_module, module in self._model.named_modules():

            if isinstance(module, ClusterAwareBatchNorm.ClusterAwareBatchNorm2d):
                self._adapt_module_names.append(name_module)
                for name_parameter, parameter in module.named_parameters():
                    # print('[info] parameter:', name_parameter)
                    if "source_rate" in name_parameter:
                        adapt_params.append(parameter)
                        adapt_param_names.append(f"{name_module}.{name_parameter}")
        print("[info] adapt_param_names:", adapt_param_names)
        assert (
            len(self._adapt_module_names) > 0
        ), "TENT needs some adaptable model parameters."
        self.adapt_param_names = adapt_param_names
        # print(parameter)
        return adapt_params, adapt_param_names

    def convert_ClusterAwareBatchNorm2d(self, module: nn.Module, **kwargs):
        """
        Recursively convert all BatchNorm to Lisc2d.
        """
        self.norm_name = "cn"
        module_output = module
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            # if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d))and layer_control == True:
            ClusterAwareBatchNorm2d = ClusterAwareBatchNorm.ClusterAwareBatchNorm2d
            module_output = ClusterAwareBatchNorm2d(
                num_channels=module.num_features,
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
            )
            module_output._bn = copy.deepcopy(module)

        for name, child in module.named_children():
            module_output.add_module(
                name, self.convert_ClusterAwareBatchNorm2d(child, **kwargs)
            )
        del module
        return module_output

    def one_adapt_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        timer: Timer,
        random_seed: int = None,
    ):
        """adapt the model in one step."""
        print("[info] adapt the model in one step.")
        # self.verify_optimizer_params(optimizer, self.adapt_param_names)

        with timer("forward"):

            with fork_rng_with_seed(random_seed):
                # with torch.no_grad():
                #     model.eval()
                #     y_hat = model(batch._x)
                model.train()
                self._meta_conf.step += 1
                y_hat_train = model(batch._x)
                loss = utils.softmax_entropy(y_hat_train).mean(0)

        with timer("backward"):
            # loss.backward()
            grads = dict(
                (name, param.grad.clone().detach())
                for name, param in model.named_parameters()
                if param.grad is not None
            )
            # optimizer.step()
            # optimizer.zero_grad()
        return {
            "optimizer": copy.deepcopy(optimizer).state_dict(),
            # "loss": 0,
            "loss": loss,
            "grads": grads,
            "yhat": y_hat_train,
        }

    def run_multiple_steps(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        model_selection_method: BaseSelection,
        nbsteps: int,
        timer: Timer,
        random_seed: int = None,
    ):
        for step in range(1, nbsteps + 1):
            adaptation_result = self.one_adapt_step(
                model,
                optimizer,
                batch,
                timer,
                random_seed=random_seed,
            )

            model_selection_method.save_state(
                {
                    "model": copy.deepcopy(model.state_dict()),
                    "step": step,
                    "lr": self._meta_conf.lr,
                    **adaptation_result,
                },
                current_batch=batch,
            )

    def adapt_and_eval(
        self,
        episodic: bool,
        metrics: Metrics,
        model_selection_method: BaseSelection,
        current_batch: Batch,
        previous_batches: List[Batch],
        logger: Logger,
        timer: Timer,
        domain_indices=None,
    ):
        """The key entry of test-time adaptation."""
        # some simple initialization.
        log = functools.partial(logger.log, display=self._meta_conf.debug)
        if episodic:
            log("\treset model to initial state during the test time.")
            self.reset()

        log(f"\tinitialize selection method={model_selection_method.name}.")
        model_selection_method.initialize()

        # evaluate the per batch pre-adapted performance. Different with no adaptation.
        # # if self._meta_conf.record_preadapted_perf:
        # with timer("evaluate_preadapted_performance"):
        #     with torch.no_grad():
        #         self._model.eval()
        #         yhat = self._model(current_batch._x)
        # self._model.train()
        #         metrics.eval_auxiliary_metric(
        #             current_batch._y, yhat, metric_name="preadapted_accuracy_top1"
        #         )
        # print(self._model)
        with timer("test_time_adaptation"):
            nbsteps = self._get_adaptation_steps(index=len(previous_batches))
            log(f"\tadapt the model for {nbsteps} steps with lr={self._meta_conf.lr}.")
            self.run_multiple_steps(
                model=self._model,
                optimizer=self._optimizer,
                batch=current_batch,
                model_selection_method=model_selection_method,
                nbsteps=nbsteps,
                timer=timer,
                random_seed=self._meta_conf.seed,
            )

        # select the optimal checkpoint, and return the corresponding prediction.
        with timer("select_optimal_checkpoint"):
            optimal_state = model_selection_method.select_state()
            log(
                f"\tselect the optimal model ({optimal_state['step']}-th step and lr={optimal_state['lr']}) for the current mini-batch.",
            )

            self._model.load_state_dict(optimal_state["model"])
            model_selection_method.clean_up()

            if self._oracle_model_selection:
                # oracle model selection needs to save steps
                self.oracle_adaptation_steps.append(optimal_state["step"])
                # update optimizer.
                self._optimizer.load_state_dict(optimal_state["optimizer"])
        # apply_grad_cam(
        #     model=self._model,
        #     # input_tensor=current_batch._x,
        #     target_layers=[self._model.layer3],
        #     # target_layers,  # Update with appropriate layers
        # )
        with timer("evaluate_adaptation_result"):
            # metrics.eval(current_batch._y, yhat)
            metrics.eval(current_batch._y, optimal_state["yhat"])
            if self._meta_conf.base_data_name in ["waterbirds"]:
                self.tta_loss_computer.loss(
                    optimal_state["yhat"],
                    current_batch._y,
                    current_batch._g,
                    is_training=False,
                )
        # Extract BN statistics and compute KL divergence
        # self.hook_extract_bn(self._model, self._meta_conf.step, current_batch)
        
        # apply_grad_cam(
        #     model=self._model,
        #     # input_tensor=current_batch._x,
        #     target_layers=[self._model.layer3],
        #     # target_layers,  # Update with appropriate layers
        # )
        # stochastic restore part of model parameters if enabled.
        if self._meta_conf.stochastic_restore_model:
            self.stochastic_restore()

    def diversity_score(self, model: nn.Module, batch: Batch):
        print("[info] diversity_score")

        model.eval()
        with torch.no_grad():
            print("[info] get feature map")

            x = model.conv1(batch._x)
        print("[info] get diversity_score")

        b, c, h, w = x.size()
        try:
            source_mu = model.bn1._bn.running_mean.view(1, c, 1, 1)
            source_sigma2 = model.bn1._bn.running_var.view(1, c, 1, 1)
        except:
            source_mu = model.bn1.running_mean.view(1, c, 1, 1)
            source_sigma2 = model.bn1.running_var.view(1, c, 1, 1)
        sigma2_b, mu_b = torch.var_mean(x, dim=[0, 2, 3], keepdim=True)

        # change the shape of source_mu and mu_b same as x
        mu_b = mu_b.repeat(b, 1, 1, 1)
        sigma2_b = sigma2_b.repeat(b, 1, 1, 1)
        source_mu = source_mu.repeat(b, 1, 1, 1)
        source_sigma2 = source_sigma2.repeat(b, 1, 1, 1)

        dsi = source_mu - x
        dti = mu_b - x
        dst = source_mu - mu_b

        cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        similarity = cos_similarity(dsi.view(b, c, -1), dst.view(b, c, -1)).mean(1)

        # similarity = similarity.cpu().detach().numpy()
        # similarity = similarity
        curve = torch.arccos(similarity)
        diversity_score = torch.std(curve)


        return diversity_score

    def hook_extract_bn(self, model: nn.Module, step: int, batch: Batch):
        """
        Hook to extract BN statistics and compute KL divergence between test-time and source BNs.
        """
        bn_stats = []

        def hook(module, input, output):
            # print(input)
            # if isinstance(module, ClusterAwareBatchNorm.ClusterAwareBatchNorm2d):
            source_mean = module._bn.running_mean
            source_var = module._bn.running_var
            test_mean = input[0].mean([0, 2, 3])
            test_var = input[0].var([0, 2, 3], unbiased=False)
            kl_div = self.kl_divergence(source_mean, source_var, test_mean, test_var)
            print(f"KL divergence: {kl_div}")
            bn_stats.append(kl_div)

        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, ClusterAwareBatchNorm.ClusterAwareBatchNorm2d):
                hooks.append(module.register_forward_hook(hook))

        self._model.eval()
        with torch.no_grad():
            _ = self._model(batch._x)

        for h in hooks:
            h.remove()

        self.save_bn_stats(bn_stats, step)


    def kl_divergence(self, mean1, var1, mean2, var2):
        """
        Compute the KL divergence between two multivariate normal distributions.
        """
        epsilon = 1e-6  # Small constant for numerical stability
        # std1 = torch.sqrt(var1 + epsilon)
        # std2 = torch.sqrt(var2 + epsilon)
        # var1 = torch.max(var1, torch.tensor(epsilon))
        # var2 = torch.max(var2,torch.tensor(epsilon))

        # kl_div = torch.log(var2 / var1) + (var1 + (mean1 - mean2).pow(2)) / (2 * var2) - 0.5
        # return kl_div.sum().item()
        mean_diff = (var1 - var2).cpu().numpy().flatten()
        # mean_diff_norm = torch.norm(mean_diff, p=2)
        mean_diff_norm = np.linalg.norm(mean_diff, ord=2)
        # cov1_sqrt = torch.linalg.cholesky(var1)
        # cov_prod_sqrt = torch.linalg.cholesky(torch.mm(torch.mm(cov1_sqrt, var2), cov1_sqrt))

        # trace_term = torch.trace(var1 + var2 - 2 * cov_prod_sqrt)

        # wasserstein_dist = torch.sqrt(mean_diff_norm**2 + trace_term)
        return mean_diff_norm.item()

    def save_bn_stats(self, bn_stats, step):
        """
        Save the BN statistics to a file, each BN stored separately.
        """
        path = "./bn_stats"
        path = os.path.join(path, self._meta_conf.model_adaptation_method)
        os.makedirs(path, exist_ok=True)

        bn_stats_dict = {}
        for i, bn_stat in enumerate(bn_stats):
            bn_stats_dict[f"bn_stats_{i}_step_{step}"] = bn_stat

        # save_path = os.path.join(path, f"bn_stats_step_{step}.json")
        # # with open(save_path, "w") as f:
        # #     json.dump(bn_stats_dict, f, indent=4)

    @property
    def name(self):
        return "DYN"


#