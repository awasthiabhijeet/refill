import datetime
import logging
import math
import os
import re
import time
import traceback
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, Type

from allennlp.common.util import int_to_device

import torch
import torch.distributed as dist
from torch.cuda import amp
import torch.optim.lr_scheduler
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_

from allennlp.common import Lazy, Registrable, Tqdm
from allennlp.common import util as common_util
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.data import DataLoader, TensorDict
from allennlp.models.model import Model
from allennlp.training import util as training_util
from allennlp.training.callbacks import TrainerCallback, SanityChecksCallback, ConsoleLoggerCallback
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer import Trainer, GradientDescentTrainer

logger = logging.getLogger(__name__)


@Trainer.register("finetuner", constructor="from_partial_objects")
class FineTuner(GradientDescentTrainer):
    def __init__(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
        data_loader: DataLoader,
        patience: Optional[int] = None,
        validation_metric: Union[str, List[str]] = "-loss",
        validation_data_loader: DataLoader = None,
        num_epochs: int = 20,
        serialization_dir: Optional[str] = None,
        checkpointer: Checkpointer = None,
        cuda_device: Optional[Union[int, torch.device]] = None,
        grad_norm: Optional[float] = None,
        grad_clipping: Optional[float] = None,
        learning_rate_scheduler: Optional[LearningRateScheduler] = None,
        momentum_scheduler: Optional[MomentumScheduler] = None,
        moving_average: Optional[MovingAverage] = None,
        callbacks: List[TrainerCallback] = None,
        distributed: bool = False,
        local_rank: int = 0,
        world_size: int = 1,
        num_gradient_accumulation_steps: int = 1,
        use_amp: bool = False,
        enable_default_callbacks: bool = True,
        run_sanity_checks: bool = True,
    ) -> None:

        super().__init__(
                        model,
                        optimizer,
                        data_loader,
                        patience,
                        validation_metric,
                        validation_data_loader,
                        num_epochs,
                        serialization_dir,
                        checkpointer,
                        cuda_device,
                        grad_norm,
                        grad_clipping,
                        learning_rate_scheduler,
                        momentum_scheduler,
                        moving_average,
                        callbacks,
                        distributed,
                        local_rank,
                        world_size,
                        num_gradient_accumulation_steps,
                        use_amp,
                        enable_default_callbacks,
                        run_sanity_checks
                    )
    @classmethod
    def from_partial_objects(
        cls,
        model: Model,
        serialization_dir: str,
        data_loader: DataLoader,
        validation_data_loader: DataLoader = None,
        local_rank: int = 0,
        patience: int = None,
        validation_metric: Union[str, List[str]] = "-loss",
        num_epochs: int = 20,
        cuda_device: Optional[Union[int, torch.device]] = None,
        grad_norm: float = None,
        grad_clipping: float = None,
        distributed: bool = False,
        world_size: int = 1,
        num_gradient_accumulation_steps: int = 1,
        use_amp: bool = False,
        no_grad: List[str] = None,
        optimizer: Lazy[Optimizer] = Lazy(Optimizer.default),
        learning_rate_scheduler: Lazy[LearningRateScheduler] = None,
        momentum_scheduler: Lazy[MomentumScheduler] = None,
        moving_average: Lazy[MovingAverage] = None,
        checkpointer: Lazy[Checkpointer] = Lazy(Checkpointer),
        callbacks: List[Lazy[TrainerCallback]] = None,
        enable_default_callbacks: bool = True,
        run_sanity_checks: bool = True,
        pretrained_ckpt: str = None,
    ) -> "Trainer":
        """
        This method exists so that we can have a documented method to construct this class using
        `FromParams`. If you are not using `FromParams` or config files, you can safely ignore this
        method.

        The reason we can't just use `__init__` with `FromParams` here is because there are
        sequential dependencies to this class's arguments.  Anything that has a `Lazy[]` type
        annotation needs something from one of the non-`Lazy` arguments.  The `Optimizer` needs to
        have the parameters from the `Model` before it's constructed, and the `Schedulers` need to
        have the `Optimizer`. Because of this, the typical way we construct things `FromParams`
        doesn't work, so we use `Lazy` to allow for constructing the objects sequentially.

        If you're not using `FromParams`, you can just construct these arguments in the right order
        yourself in your code and call the constructor directly.
        """
        if pretrained_ckpt is not None:
            print(f'\n***** Loading from {pretrained_ckpt} ******\n')
            pretrained_weights = torch.load(pretrained_ckpt, "cpu")
            model.load_state_dict(pretrained_weights, strict=False)

        if cuda_device is None:
            from torch import cuda

            if cuda.device_count() > 0:
                cuda_device = 0
            else:
                cuda_device = -1

        check_for_gpu(cuda_device)
        if cuda_device >= 0:
            # Moving model to GPU here so that the optimizer state gets constructed on
            # the right device.
            model = model.cuda(cuda_device)

        if no_grad:
            for name, parameter in model.named_parameters():
                if any(re.search(regex, name) for regex in no_grad):
                    parameter.requires_grad_(False)

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer_ = optimizer.construct(model_parameters=parameters)

        common_util.log_frozen_and_tunable_parameter_names(model)

        batches_per_epoch: Optional[int]
        try:
            batches_per_epoch = len(data_loader)
            batches_per_epoch = math.ceil(batches_per_epoch / num_gradient_accumulation_steps)
        except TypeError:
            batches_per_epoch = None

        moving_average_ = (
            None if moving_average is None else moving_average.construct(parameters=parameters)
        )
        learning_rate_scheduler_ = (
            None
            if learning_rate_scheduler is None
            else learning_rate_scheduler.construct(
                optimizer=optimizer_, num_epochs=num_epochs, num_steps_per_epoch=batches_per_epoch
            )
        )
        momentum_scheduler_ = (
            None
            if momentum_scheduler is None
            else momentum_scheduler.construct(optimizer=optimizer_)
        )
        checkpointer_ = checkpointer.construct(serialization_dir=serialization_dir)

        callbacks_: List[TrainerCallback] = []
        for callback_ in callbacks or []:
            callbacks_.append(callback_.construct(serialization_dir=serialization_dir))

        return cls(
            model,
            optimizer_,
            data_loader,
            patience=patience,
            validation_metric=validation_metric,
            validation_data_loader=validation_data_loader,
            num_epochs=num_epochs,
            serialization_dir=serialization_dir,
            cuda_device=cuda_device,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping,
            learning_rate_scheduler=learning_rate_scheduler_,
            momentum_scheduler=momentum_scheduler_,
            checkpointer=checkpointer_,
            moving_average=moving_average_,
            callbacks=callbacks_,
            distributed=distributed,
            local_rank=local_rank,
            world_size=world_size,
            num_gradient_accumulation_steps=num_gradient_accumulation_steps,
            use_amp=use_amp,
            enable_default_callbacks=enable_default_callbacks,
            run_sanity_checks=run_sanity_checks,
        )


DEFAULT_CALLBACKS: Tuple[Type[TrainerCallback]] = (ConsoleLoggerCallback,)
"""
The default callbacks used by `GradientDescentTrainer`.
"""