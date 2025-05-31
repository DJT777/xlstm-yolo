# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING, colorstr
import torch.nn as nn

try:
    from torch.utils.tensorboard import SummaryWriter
    assert not TESTS_RUNNING
    assert SETTINGS["tensorboard"] is True
    WRITER = None
    PREFIX = colorstr("TensorBoard: ")
    import warnings
    from copy import deepcopy
    from ultralytics.utils.torch_utils import de_parallel, torch
except (ImportError, AssertionError, TypeError, AttributeError):
    SummaryWriter = None


def _log_scalars(scalars, step=0):
    if WRITER:
        for k, v in scalars.items():
            WRITER.add_scalar(k, v, step)
        WRITER.flush()


def _log_tensorboard_graph(trainer):
    imgsz = trainer.args.imgsz
    imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    p = next(trainer.model.parameters())
    im = torch.zeros((1, 3, *imgsz), device=p.device, dtype=p.dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
        try:
            trainer.model.eval()
            WRITER.add_graph(torch.jit.trace(de_parallel(trainer.model), im, strict=False), [])
            LOGGER.info(f"{PREFIX}model graph added ✅")
            return
        except Exception:
            try:
                model = deepcopy(de_parallel(trainer.model))
                model.eval()
                model = model.fuse(verbose=False)
                for m in model.modules():
                    if hasattr(m, "export"):
                        m.export = True
                        m.format = "torchscript"
                model(im)
                WRITER.add_graph(torch.jit.trace(model, im, strict=False), [])
                LOGGER.info(f"{PREFIX}model graph added ✅")
            except Exception as e:
                LOGGER.warning(f"{PREFIX}graph failure {e}")


def on_pretrain_routine_start(trainer):
    if SummaryWriter:
        try:
            global WRITER
            WRITER = SummaryWriter(str(trainer.save_dir))
            LOGGER.info(f"{PREFIX}TensorBoard logging to {trainer.save_dir}")
        except Exception as e:
            LOGGER.warning(f"{PREFIX}init failure {e}")

# Activation monitoring globals
activations = {}
capture_activations = False
hooks_setup = False


# def get_activation_hook(name):
#     """Returns a hook that stores the layer's output (handling Tensors, lists, tuples) if capture_activations is True."""
#     def hook_fn(module, input, output):
#         if not capture_activations:
#             return
#         # single Tensor
#         if isinstance(output, torch.Tensor):
#             activations[name] = output.detach()
#         # list or tuple of Tensors
#         elif isinstance(output, (list, tuple)):
#             for idx, item in enumerate(output):
#                 if isinstance(item, torch.Tensor):
#                     activations[f"{name}.{idx}"] = item.detach()
#         # other types are ignored
#     return hook_fn


# def register_activation_hooks(trainer):
#     global hooks_setup
#     if not hooks_setup and SummaryWriter:
#         for name, module in trainer.model.named_modules():
#             module.register_forward_hook(get_activation_hook(name))
#         hooks_setup = True


def check_anomalies(trainer, step):
    """Scan activations and gradients for NaN/Inf and log counts."""
    if not WRITER:
        return
    # activations
    for name, act in activations.items():
        nan = int(torch.isnan(act).sum().item())
        inf = int(torch.isinf(act).sum().item())
        if nan or inf:
            LOGGER.warning(f"Anomaly act {name}: NaN={nan}, Inf={inf}")
        WRITER.add_scalar(f"anomaly/activations/{name}_nan", nan, step)
        WRITER.add_scalar(f"anomaly/activations/{name}_inf", inf, step)
    # gradients
    for name, p in trainer.model.named_parameters():
        if p.grad is not None:
            nan = int(torch.isnan(p.grad).sum().item())
            inf = int(torch.isinf(p.grad).sum().item())
            if nan or inf:
                LOGGER.warning(f"Anomaly grad {name}: NaN={nan}, Inf={inf}")
            WRITER.add_scalar(f"anomaly/gradients/{name}_nan", nan, step)
            WRITER.add_scalar(f"anomaly/gradients/{name}_inf", inf, step)
    WRITER.flush()


def on_train_epoch_start(trainer):
    trainer.batch_idx = 0


# def on_train_batch_start(trainer):
#     register_activation_hooks(trainer)
#     global capture_activations
#     capture_activations = True
#     trainer.batch_idx = getattr(trainer, 'batch_idx', 0) + 1


def log_histograms(trainer):
    if WRITER and capture_activations:
        batches = len(getattr(trainer, 'train_loader', []))
        step = trainer.epoch * batches + getattr(trainer, 'batch_idx', 0)
        for name, act in activations.items():
            WRITER.add_histogram(f'activations/{name}', act, step)
        for name, p in trainer.model.named_parameters():
            if p.grad is not None:
                WRITER.add_histogram(f'gradients/{name}', p.grad, step)
        check_anomalies(trainer, step)
        WRITER.flush()
        activations.clear()


def on_train_start(trainer):
    if WRITER:
        _log_tensorboard_graph(trainer)


def on_train_epoch_end(trainer):
    _log_scalars(trainer.label_loss_items(trainer.tloss, prefix="train"), trainer.epoch + 1)


def on_fit_epoch_end(trainer):
    _log_scalars(trainer.metrics, trainer.epoch + 1)

# Assemble callbacks dict
callbacks = ({
    "on_pretrain_routine_start": on_pretrain_routine_start,
    "on_train_start":             on_train_start,
    "on_train_epoch_start":       on_train_epoch_start,
    #"on_train_batch_start":       on_train_batch_start,
    #"on_train_batch_end":         log_histograms,
    "on_train_epoch_end":         on_train_epoch_end,
    "on_fit_epoch_end":           on_fit_epoch_end,
} if SummaryWriter else {})
