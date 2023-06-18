
import importlib
import os

from optimizers.lr_schedulers.fairseq_lr_scheduler import FairseqLRScheduler
from .inverse_square_root_schedule import InverseSquareRootSchedule

LR_SCHEDULER_REGISTRY = {}


def build_lr_scheduler(args, optimizer):
    return LR_SCHEDULER_REGISTRY[args.lr_scheduler](args, optimizer)


def register_lr_scheduler(name):
    """Decorator to register a new LR scheduler."""

    def register_lr_scheduler_cls(cls):
        if name in LR_SCHEDULER_REGISTRY:
            raise ValueError('Cannot register duplicate LR scheduler ({})'.format(name))
        if not issubclass(cls, FairseqLRScheduler):
            raise ValueError('LR Scheduler ({}: {}) must extend FairseqLRScheduler'.format(name, cls.__name__))
        LR_SCHEDULER_REGISTRY[name] = cls
        return cls

    return register_lr_scheduler_cls


# automatically import any Python files in the optimizers/lr_schedulers/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('optimizers.lr_schedulers.' + module)
