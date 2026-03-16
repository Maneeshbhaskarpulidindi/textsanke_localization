import os
import time
import math
from datetime import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.optim import lr_scheduler
from util.shedule import FixLR

from dataset.total_text import TotalText
from dataset.synth_text import SynthText
from dataset.custom_text import CustomTextDataset
from network.loss import TextLoss
from network.textnet import TextNet
from util.augmentation import BaseTransform, Augmentation
from util.config import config as cfg, update_config, print_config
from util.misc import AverageMeter
from util.misc import mkdirs, to_device
from util.option import BaseOptions
from util.visualize import visualize_network_output
from util.summary import LogSummary

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

# global vars
lr = None
train_step = 0
best_val_loss = float("inf")   # track best validation loss


from torch.optim.lr_scheduler import LambdaLR
import math

def warmup_cosine_lr(optimizer, base_lr, warmup_epochs, max_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / float(max_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def to_float(val):
    return val.item() if hasattr(val, 'item') else float(val)


def save_model(model, epoch, lr, optimizer, name=None):
    save_dir = os.path.join(cfg.save_dir, cfg.exp_name)
    if not os.path.exists(save_dir):
        mkdirs(save_dir)

    if name is None:
        save_path = os.path.join(save_dir, f"textsnake_{model.backbone_name}_{epoch}.pth")
    else:
        save_path = os.path.join(save_dir, f"{name}.pth")

    print(f"Saving to {save_path}.")
    state_dict = {
        "lr": lr,
        "epoch": epoch,
        "model": model.state_dict() if not cfg.mgpu else model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state_dict, save_path)


def load_model(model, model_path, optimizer=None):
    print("Loading from {}".format(model_path))
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict["model"])
    if optimizer is not None and "optimizer" in state_dict:
        optimizer.load_state_dict(state_dict["optimizer"])
    if "epoch" in state_dict:
        return state_dict["epoch"]
    return None


# def train(model, train_loader, criterion, scheduler, optimizer, epoch, logger, scaler):
#     global train_step
#     losses = AverageMeter()
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     end = time.time()
#     model.train()
#     scheduler.step()

#     print(f"Epoch: {epoch} : LR = {scheduler.get_last_lr()[0]}")

#     for i, (img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta) in enumerate(train_loader):
#         data_time.update(time.time() - end)
#         train_step += 1

#         img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map = to_device(
#             img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map
#         )

#         optimizer.zero_grad()

#         # --- AMP forward + backward ---
#         with torch.amp.autocast('cuda', enabled=cfg.cuda):
#             output = model(img)
#             tr_loss, tcl_loss, sin_loss, cos_loss, radii_loss = criterion(
#                 output, tr_mask, tcl_mask, sin_map, cos_map, radius_map, train_mask
#             )
#             loss = tr_loss + tcl_loss + sin_loss + cos_loss + radii_loss

#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         # logging
#         losses.update(loss.item())
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if cfg.viz and i % cfg.viz_freq == 0:
#             visualize_network_output(output, tr_mask, tcl_mask, mode="train")

#         if i % cfg.display_freq == 0:
#             print(
#                 "({:d} / {:d}) - Loss: {:.4f} - tr_loss: {:.4f} - tcl_loss: {:.4f} - sin_loss: {:.4f} - cos_loss: {:.4f} - radii_loss: {:.4f}".format(
#                     i, len(train_loader),
#                     to_float(loss), to_float(tr_loss), to_float(tcl_loss),
#                     to_float(sin_loss), to_float(cos_loss), to_float(radii_loss)
#                 )
#             )

#         if i % cfg.log_freq == 0:
#             logger.write_scalars({
#                 "loss": to_float(loss),
#                 "tr_loss": to_float(tr_loss),
#                 "tcl_loss": to_float(tcl_loss),
#                 "sin_loss": to_float(sin_loss),
#                 "cos_loss": to_float(cos_loss),
#                 "radii_loss": to_float(radii_loss)
#             }, tag="train", n_iter=train_step)

#     if epoch % cfg.save_freq == 0:
#         save_model(model, epoch, scheduler.get_last_lr()[0], optimizer)

#     print("Training Loss: {}".format(losses.avg))
def train(model, train_loader, criterion, scheduler, optimizer, epoch, logger, scaler):
    global train_step
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()
    model.train()

    # just log LR at epoch start (don't step scheduler here)
    cur_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch: {epoch} : LR = {cur_lr}")

    for i, (img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, instance_labels, meta) in enumerate(train_loader):
        data_time.update(time.time() - end)
        train_step += 1

        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, instance_labels = to_device(
            img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, instance_labels
        )

        optimizer.zero_grad(set_to_none=True)

        # --- AMP forward + backward ---
        with torch.amp.autocast('cuda', enabled=cfg.cuda):
            prediction, embedding = model(img)
            loss, loss_dict = criterion(
                prediction, embedding,
                tr_mask, tcl_mask, sin_map, cos_map, radius_map,
                train_mask, instance_labels
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # logging
        losses.update(float(loss))
        batch_time.update(time.time() - end)
        end = time.time()

        if cfg.viz and i % cfg.viz_freq == 0:
            visualize_network_output(prediction, tr_mask, tcl_mask, mode="train", logger=logger, n_iter=train_step)

        if i % cfg.display_freq == 0:
            print(
                "({:d} / {:d}) - Loss: {:.4f} - tr_loss: {:.4f} - tcl_loss: {:.4f} - "
                "sin_loss: {:.4f} - cos_loss: {:.4f} - radii_loss: {:.4f} - "
                "pull: {:.4f} - push: {:.4f} - embed: {:.4f}".format(
                    i, len(train_loader),
                    to_float(loss),
                    to_float(loss_dict['tr_loss']),
                    to_float(loss_dict['tcl_loss']),
                    to_float(loss_dict['sin_loss']),
                    to_float(loss_dict['cos_loss']),
                    to_float(loss_dict['radii_loss']),
                    to_float(loss_dict['pull_loss']),
                    to_float(loss_dict['push_loss']),
                    to_float(loss_dict['embed_loss']),
                )
            )

        if i % cfg.log_freq == 0:
            log_scalars = {
                "loss": to_float(loss),
            }
            for key in loss_dict:
                log_scalars[key] = to_float(loss_dict[key])
            logger.write_scalars(log_scalars, tag="train", n_iter=train_step)

    if epoch % cfg.save_freq == 0:
        # log LR from optimizer to keep it consistent
        save_model(model, epoch, optimizer.param_groups[0]['lr'], optimizer)

    print("Training Loss: {}".format(losses.avg))



def validation(model, valid_loader, criterion, epoch, logger, optimizer):
    global best_val_loss
    with torch.no_grad():
        model.eval()
        losses = AverageMeter()

        for i, (img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, instance_labels, meta) in enumerate(valid_loader):
            img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, instance_labels = to_device(
                img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, instance_labels
            )

            prediction, embedding = model(img)
            loss, loss_dict = criterion(
                prediction, embedding,
                tr_mask, tcl_mask, sin_map, cos_map, radius_map,
                train_mask, instance_labels
            )
            losses.update(loss.item())

            if cfg.viz and i % cfg.viz_freq == 0:
                visualize_network_output(prediction, tr_mask, tcl_mask, mode="val", logger=logger, n_iter=epoch)

            if i % cfg.display_freq == 0:
                print(
                    "Validation: - Loss: {:.4f} - tr: {:.4f} - tcl: {:.4f} - sin: {:.4f} - "
                    "cos: {:.4f} - radii: {:.4f} - pull: {:.4f} - push: {:.4f} - embed: {:.4f}".format(
                        loss.item(),
                        to_float(loss_dict['tr_loss']),
                        to_float(loss_dict['tcl_loss']),
                        to_float(loss_dict['sin_loss']),
                        to_float(loss_dict['cos_loss']),
                        to_float(loss_dict['radii_loss']),
                        to_float(loss_dict['pull_loss']),
                        to_float(loss_dict['push_loss']),
                        to_float(loss_dict['embed_loss']),
                    )
                )

        logger.write_scalars({"loss": losses.avg}, tag="val", n_iter=epoch)
        print("Validation Loss: {}".format(losses.avg))

        # --- Best checkpoint saving ---
# --- Best checkpoint saving ---
        if losses.avg < best_val_loss:
            best_val_loss = losses.avg
            current_lr = optimizer.param_groups[0]['lr']
            save_model(model, epoch, current_lr, optimizer, name="best")
            print(f"✨ New best model saved with loss={best_val_loss:.4f}")


# def main():
#     global lr
#     if cfg.dataset == "total-text":
#         trainset = TotalText(
#             data_root="data/total-text",
#             ignore_list=None,
#             is_training=True,
#             transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
#         )
#         valset = TotalText(
#             data_root="data/total-text",
#             ignore_list=None,
#             is_training=False,
#             transform=BaseTransform(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
#         )
#     elif cfg.dataset == "synth-text":
#         trainset = SynthText(
#             data_root="data/SynthText",
#             is_training=True,
#             transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
#         )
#         valset = None
#     else:
#         valset = None

#     train_loader = data.DataLoader(
#         trainset,
#         batch_size=cfg.batch_size,
#         shuffle=True,
#         num_workers=cfg.num_workers,
#         generator=torch.Generator(device="cuda") if cfg.cuda else None
#     )
#     val_loader = None
#     if valset:
#         val_loader = data.DataLoader(valset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

#     log_dir = os.path.join(cfg.log_dir, datetime.now().strftime("%b%d_%H-%M-%S_") + cfg.exp_name)
#     logger = LogSummary(log_dir)

#     # Model
#     model = TextNet(is_training=True, backbone=cfg.net)
#     if cfg.mgpu:
#         model = nn.DataParallel(model)
#     model = model.to(cfg.device)

#     if cfg.cuda:
#         cudnn.benchmark = True

#     if cfg.resume:
#         load_model(model, cfg.resume)

#     criterion = TextLoss()
#     lr = cfg.lr
#     optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

#     # AMP scaler
#     scaler = torch.cuda.amp.GradScaler(enabled=cfg.cuda)

#     # Scheduler
#     if cfg.dataset == "synth-text":
#         scheduler = FixLR(optimizer)
#     else:
#         # scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
#         scheduler = warmup_cosine_lr(
#                                         optimizer,
#                                         base_lr=cfg.lr,
#                                         warmup_epochs=5,           # configurable
#                                         max_epochs=cfg.max_epoch
#                                     )


#     print("Start training TextSnake.")
#     for epoch in range(cfg.start_epoch, cfg.max_epoch):
#         train(model, train_loader, criterion, scheduler, optimizer, epoch, logger, scaler)
#         if val_loader:
#             validation(model, val_loader, criterion, epoch, logger, optimizer)
#     print("End.")


def main():
    global lr

    # -------- Datasets --------
    if cfg.dataset == "total-text":
        trainset = TotalText(
            data_root="data/total-text",
            ignore_list=None,
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = TotalText(
            data_root="data/total-text",
            ignore_list=None,
            is_training=False,
            transform=BaseTransform(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
    elif cfg.dataset == "synth-text":
        trainset = SynthText(
            data_root="data/SynthText",
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None
    elif cfg.dataset == "custom":
        assert cfg.data_root is not None, \
            "Please pass --data_root <path> when using --dataset custom"
        trainset = CustomTextDataset(
            data_root=cfg.data_root,
            is_training=True,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = CustomTextDataset(
            data_root=cfg.data_root,
            is_training=False,
            transform=BaseTransform(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
    else:
        valset = None

    train_loader = data.DataLoader(
        trainset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        generator=torch.Generator(device="cuda") if cfg.cuda else None
    )

    val_loader = None
    if valset:
        val_loader = data.DataLoader(
            valset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers
        )

    # -------- Logging --------
    log_dir = os.path.join(cfg.log_dir, datetime.now().strftime("%b%d_%H-%M-%S_") + cfg.exp_name)
    logger = LogSummary(log_dir)

    # -------- Model --------
    model = TextNet(is_training=True, backbone=cfg.net)
    if cfg.mgpu:
        model = nn.DataParallel(model)
    model = model.to(cfg.device)

    if cfg.cuda:
        cudnn.benchmark = True

    # -------- Optimizer / Loss / AMP --------
    criterion = TextLoss(delta_v=cfg.delta_v, delta_d=cfg.delta_d, lambda_embed=cfg.lambda_embed)
    lr = cfg.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.cuda)

    if cfg.resume:
        start_epoch = load_model(model, cfg.resume, optimizer)
        if start_epoch is not None:
            cfg.start_epoch = start_epoch + 1
            print(f"Resuming from epoch {start_epoch}, next epoch is {cfg.start_epoch}")

    # -------- Scheduler (epoch-based; step AFTER each epoch) --------
    if cfg.dataset == "synth-text":
        scheduler = FixLR(optimizer)  # constant LR (safe to call step() per epoch)
    else:
        scheduler = warmup_cosine_lr(
            optimizer,
            base_lr=cfg.lr,
            warmup_epochs=5,        # tweak if needed
            max_epochs=cfg.max_epoch
        )
    
    # Catch up the scheduler
    for _ in range(cfg.start_epoch):
        scheduler.step()

    # -------- Train Loop --------
    print("Start training TextSnake.")
    for epoch in range(cfg.start_epoch, cfg.max_epoch):
        train(model, train_loader, criterion, scheduler, optimizer, epoch, logger, scaler)
        if val_loader:
            validation(model, val_loader, criterion, epoch, logger, optimizer)

        # ✅ Step the scheduler AFTER optimizer updates in this epoch
        scheduler.step()

    print("End.")


if __name__ == "__main__":
    option = BaseOptions()
    args = option.initialize()
    update_config(cfg, args)
    print_config(cfg)
    main()
