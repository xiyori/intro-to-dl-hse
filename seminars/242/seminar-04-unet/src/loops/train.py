import torch
import numpy as np

from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter
from typing import Dict
from collections import defaultdict
from tqdm.auto import tqdm

from ..utils import checkpoint, image_grid


def get_n_best_metric(metrics, n_best):
    return np.partition(metrics[:-1], -n_best)[-n_best]


def set_requires_grad(module: nn.Module, requires_grad: bool):
    module.train(requires_grad)
    for param in module.parameters():
        param.requires_grad = requires_grad


def losses(generator: nn.Module, discriminator: nn.Module,
           input: Tensor, target: Tensor, criterion: Dict,
           dis_loss_coef: float, skip_dis_step: bool = True,
           gan_real_label: float = 1.0):
    output = generator(input)

    concat_output = torch.cat((output, input), 1)
    concat_target = torch.cat((target, input), 1)

    set_requires_grad(discriminator, False)

    super_losses = {key: (coef, loss_fn(output, target)) for key, (coef, loss_fn) in criterion.items()}

    gen_loss = sum(map(lambda x: x[0] * x[1], super_losses.values())) + dis_loss_coef * \
               ((discriminator(concat_output) - gan_real_label) ** 2).mean()

    concat_output = concat_output.detach()

    if not skip_dis_step:
        set_requires_grad(discriminator, True)

    dis_loss = (discriminator(concat_output) ** 2 +
                (discriminator(concat_target) - gan_real_label) ** 2).mean()

    return output, super_losses, gen_loss, dis_loss


def train_gan(config, train_loader, valid_loader,
              generator, discriminator, gen_optimizer, dis_optimizer,
              gen_scheduler=None, dis_scheduler=None, resume=False, progress="epochs"):
    criterion = config.loss
    metric = config.metric

    if isinstance(criterion, nn.Module):
        criterion = {config.loss_name: criterion}
    if isinstance(metric, nn.Module):
        metric = {config.valid_metric_name: metric}

    if resume:
        start_epoch, valid_metric_history = checkpoint.load(
            config.run_name, generator, gen_optimizer, gen_scheduler,
            discriminator, dis_optimizer, dis_scheduler
        )
    else:
        start_epoch = 0
        valid_metric_history = []

    writer = SummaryWriter(log_dir=f"resources/tensorboard/{config.run_name}")
    global_step = start_epoch * len(train_loader)

    dis_loss = 1
    skip_dis_step = False
    steps_skipped = 0
    steps_waited = 0

    epoch_iter = range(start_epoch + 1, config.num_epochs + 1)
    if progress == "epochs":
        epoch_iter = tqdm(epoch_iter, desc="Epoch")

    for epoch in epoch_iter:
        generator.train()
        discriminator.train()

        train_iter = train_loader
        if progress == "samples":
            train_iter = tqdm(train_iter, desc=f"Train {epoch}/{config.num_epochs}")

        for input, target in train_iter:
            input = input.to(config.device)
            target = target.to(config.device)

            gen_optimizer.zero_grad()
            dis_optimizer.zero_grad()

            if dis_loss < config.min_dis_loss:
                if not skip_dis_step:
                    steps_waited += 1
                if steps_waited >= config.steps_to_wait:
                    skip_dis_step = True
                    steps_skipped = 0
            elif skip_dis_step:
                steps_skipped += 1
                if steps_skipped >= config.steps_to_skip:
                    skip_dis_step = False
                    steps_waited = 0

            output, super_losses, gen_loss, dis_loss = losses(
                generator, discriminator, input, target,
                criterion, config.dis_loss_coef, skip_dis_step,
                config.gan_real_label
            )

            if (torch.isnan(gen_loss) or torch.isinf(gen_loss) or
                    torch.isnan(dis_loss) or torch.isinf(dis_loss)):
                print(gen_loss, dis_loss)
                writer.close()
                return "gradient explosion"

            gen_loss.backward()
            if config.gen_grad_clip_threshold is not None:
                nn.utils.clip_grad_norm_(generator.parameters(), config.gen_grad_clip_threshold)
            gen_optimizer.step()

            if not skip_dis_step:
                dis_loss.backward()
                if config.dis_grad_clip_threshold is not None:
                    nn.utils.clip_grad_norm_(discriminator.parameters(), config.dis_grad_clip_threshold)
                dis_optimizer.step()

            log = {"train_" + key: value.item() for key, (_, value) in super_losses.items()}
            log["train_generator_loss"] = gen_loss.item()
            log["train_discriminator_loss"] = dis_loss.item()
            log["discriminator_enabled"] = int(not skip_dis_step)
            with torch.no_grad():
                for key, m in metric.items():
                    log["train_" + key] = m(output, target).item()

            for key, value in log.items():
                writer.add_scalar(key, value, global_step)
            global_step += 1

        valid_log = defaultdict(float)
        generator.eval()
        discriminator.eval()

        valid_iter = valid_loader
        if progress == "samples":
            valid_iter = tqdm(valid_iter, desc=f"Valid {epoch}/{config.num_epochs}")

        image_samples = None
        with torch.no_grad():
            for input, target in valid_iter:
                input = input.to(config.device)
                target = target.to(config.device)

                output, super_losses, gen_loss, dis_loss = losses(
                    generator, discriminator, input, target,
                    criterion, config.dis_loss_coef,
                    gan_real_label=config.gan_real_label
                )
                image_samples = image_grid(input, output, target, num_images=4)

                valid_log["valid_generator_loss"] += gen_loss.item()
                valid_log["valid_discriminator_loss"] += dis_loss.item()
                for key, (_, value) in super_losses.items():
                    valid_log["valid_" + key] += value.item()
                for key, m in metric.items():
                    valid_log["valid_" + key] += m(output, target).item()

        for key in valid_log:
            valid_log[key] /= len(valid_loader)

        valid_log = dict(valid_log)
        valid_log["generator_lr"] = gen_optimizer.param_groups[0]["lr"]
        valid_log["discriminator_lr"] = dis_optimizer.param_groups[0]["lr"]
        valid_log["epoch"] = epoch

        for key, value in valid_log.items():
            writer.add_scalar(key, value, epoch)
        if image_samples is not None:
            writer.add_image("valid_image_samples", (image_samples.cpu() / 2 + 0.5).clip(0, 1), epoch)

        valid_metric_history += [valid_log["valid_" + config.valid_metric_name]]

        if config.provide_metric_to_scheduler:
            if gen_scheduler is not None:
                gen_scheduler.step(valid_metric_history[-1])
            if dis_scheduler is not None:
                dis_scheduler.step(valid_metric_history[-1])
        else:
            if gen_scheduler is not None:
                gen_scheduler.step()
            if dis_scheduler is not None:
                dis_scheduler.step()

        if (epoch <= config.n_best_save or valid_metric_history[-1] >
                get_n_best_metric(valid_metric_history, config.n_best_save)):
            checkpoint.save_model(config.run_name, epoch, generator)

        checkpoint.save(config.run_name, epoch, valid_metric_history,
                        generator, gen_optimizer, gen_scheduler,
                        discriminator, dis_optimizer, dis_scheduler)

    print("Best valid %s:  %.3f on epoch %d" %
          (config.valid_metric_name, max(valid_metric_history), np.argmax(valid_metric_history) + 1))

    writer.close()
    return "success"


def train(config, train_loader, valid_loader,
          generator, discriminator, gen_optimizer, dis_optimizer,
          gen_scheduler=None, dis_scheduler=None, resume=False, progress="epochs"):
    return train_gan(config, train_loader, valid_loader,
                     generator, discriminator, gen_optimizer, dis_optimizer,
                     gen_scheduler, dis_scheduler, resume, progress)
