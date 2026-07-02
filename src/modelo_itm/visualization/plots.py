import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.data import Subset

from modelo_itm.config import Config
from modelo_itm.inference.uncertainty import build_uncertainty_map, predict_with_uncertainty
from modelo_itm.utils.io import ensure_dir


def save_history_plots(history, out_dir):
    if not history:
        return

    path = Path(out_dir)
    ensure_dir(path)

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].plot(epochs, [row["train_loss"] for row in history], label="Train loss", linewidth=2.0)
    axes[0, 0].plot(epochs, [row["val_loss"] for row in history], label="Val loss", linewidth=2.0)
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend(loc="best")

    axes[0, 1].plot(epochs, [row["train_sf_rmse"] for row in history], label="Train SF RMSE", linewidth=2.0)
    axes[0, 1].plot(epochs, [row["val_sf_rmse"] for row in history], label="Val SF RMSE", linewidth=2.0)
    axes[0, 1].plot(epochs, [row["train_vd_rmse"] for row in history], label="Train VD RMSE", linewidth=2.0)
    axes[0, 1].plot(epochs, [row["val_vd_rmse"] for row in history], label="Val VD RMSE", linewidth=2.0)
    axes[0, 1].set_title("RMSE")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend(loc="best")

    axes[1, 0].plot(epochs, [row["train_sf_r2"] for row in history], label="Train SF R2", linewidth=2.0)
    axes[1, 0].plot(epochs, [row["val_sf_r2"] for row in history], label="Val SF R2", linewidth=2.0)
    axes[1, 0].plot(epochs, [row["train_vd_r2"] for row in history], label="Train VD R2", linewidth=2.0)
    axes[1, 0].plot(epochs, [row["val_vd_r2"] for row in history], label="Val VD R2", linewidth=2.0)
    axes[1, 0].set_title("R2")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend(loc="best")

    axes[1, 1].plot(
        epochs,
        [row.get("val_sf_uncertainty_mean", 0.0) for row in history],
        label="Val SF uncertainty",
        linewidth=2.0,
    )
    axes[1, 1].plot(
        epochs,
        [row.get("val_vd_uncertainty_mean", 0.0) for row in history],
        label="Val VD uncertainty",
        linewidth=2.0,
    )
    axes[1, 1].set_title("Average prediction uncertainty")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("0-1")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend(loc="best")

    plt.tight_layout()
    plt.savefig(path / "training_curves.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(
        epochs,
        [row.get("val_sf_confidence_mean", 0.0) for row in history],
        label="Val SF confidence",
        linewidth=2.0,
    )
    axes[0].plot(
        epochs,
        [row.get("val_vd_confidence_mean", 0.0) for row in history],
        label="Val VD confidence",
        linewidth=2.0,
    )
    axes[0].set_title("Confidence")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("0-1")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(
        epochs,
        [row.get("val_sf_uncertainty_p95", 0.0) for row in history],
        label="Val SF uncertainty p95",
        linewidth=2.0,
    )
    axes[1].plot(
        epochs,
        [row.get("val_vd_uncertainty_p95", 0.0) for row in history],
        label="Val VD uncertainty p95",
        linewidth=2.0,
    )
    axes[1].set_title("Uncertainty p95")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("0-1")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")

    plt.tight_layout()
    plt.savefig(path / "uncertainty_curves.png")
    plt.close(fig)


def save_epoch_visuals(model, dataset, epoch, out_dir, device, cfg: Config, calibration: dict):
    if dataset is None or len(dataset) == 0:
        return

    model.eval()
    path = Path(out_dir) / "visuals"
    ensure_dir(path)

    with torch.no_grad():
        if isinstance(dataset, Subset):
            rel_idx = random.randrange(len(dataset.indices))
            base_idx = int(dataset.indices[rel_idx])
            base_dataset = dataset.dataset
        else:
            base_idx = random.randrange(len(dataset))
            base_dataset = dataset

        x, d, inj, y = base_dataset[base_idx]
        meta = base_dataset.samples[base_idx]
        pred, pred_std = predict_with_uncertainty(
            model,
            x.unsqueeze(0).to(device),
            d.unsqueeze(0).to(device),
            inj.unsqueeze(0).to(device),
            cfg.uncertainty_passes,
        )
        pred = pred[0].cpu()
        pred_std = pred_std[0].cpu()
        uncertainty_map = build_uncertainty_map(pred_std.unsqueeze(0), calibration)[0]
        confidence_map = 1.0 - uncertainty_map

        times = [0, min(30, y.shape[0] - 1), y.shape[0] - 1]
        times = list(dict.fromkeys(times))
        fig, axes = plt.subplots(8, len(times), figsize=(6 * len(times), 26))
        if len(times) == 1:
            axes = np.expand_dims(axes, axis=1)

        for col, t in enumerate(times):
            for row, data, title, cmap, vmax in [
                (0, y, "GT SF", "viridis", 1.0),
                (1, pred, "Pred SF", "viridis", 1.0),
                (2, uncertainty_map, "SF uncertainty", "inferno", 1.0),
                (3, confidence_map, "SF confidence", "viridis", 1.0),
            ]:
                im = axes[row, col].imshow(data[t, 0], cmap=cmap, vmin=0, vmax=vmax)
                axes[row, col].set_title(f"{title} t={t}")
                divider = make_axes_locatable(axes[row, col])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(im, cax=cax)

            vmax_vd = max(y[:, 1].max().item(), pred[:, 1].max().item()) + 1e-6
            for row, data, title, cmap in [
                (4, y, "GT VD", "plasma"),
                (5, pred, "Pred VD", "plasma"),
            ]:
                im = axes[row, col].imshow(data[t, 1], cmap=cmap, vmin=0, vmax=vmax_vd)
                axes[row, col].set_title(f"{title} t={t}")
                divider = make_axes_locatable(axes[row, col])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(im, cax=cax)

            im = axes[6, col].imshow(uncertainty_map[t, 1], cmap="inferno", vmin=0, vmax=1.0)
            axes[6, col].set_title(f"VD uncertainty t={t}")
            divider = make_axes_locatable(axes[6, col])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)

            im = axes[7, col].imshow(confidence_map[t, 1], cmap="viridis", vmin=0, vmax=1.0)
            axes[7, col].set_title(f"VD confidence t={t}")
            divider = make_axes_locatable(axes[7, col])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)

        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f"Epoch {epoch:03d} | case={meta['case']} | layer={meta['k']}", fontsize=14)
        plt.tight_layout(rect=(0, 0, 1, 0.97))
        plt.savefig(path / f"epoch_{epoch:03d}.png")
        plt.close(fig)

        input_fig, input_axes = plt.subplots(2, 3, figsize=(16, 10))
        input_panels = [
            ("AFI", x[0], "viridis"),
            ("COH", x[1], "viridis"),
            ("PERM", x[2], "magma"),
            ("PORO", x[3], "cividis"),
        ]

        for ax, (title, data, cmap) in zip(input_axes.flatten(), input_panels):
            im = ax.imshow(data, cmap=cmap)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            input_fig.colorbar(im, cax=cax)

        inj_ax = input_axes[1, 1]
        t = np.arange(inj.shape[0])
        inj_np = inj.cpu().numpy()
        inj_ax.plot(t, inj_np[:, 0], label="Inj 1", linewidth=2.0)
        inj_ax.plot(t, inj_np[:, 1], label="Inj 2", linewidth=2.0)
        inj_ax.set_title("Real Injection Input")
        inj_ax.set_xlabel("Time")
        inj_ax.set_ylabel("Normalized value")
        inj_ax.grid(alpha=0.3)
        inj_ax.legend(loc="best")

        meta_ax = input_axes[1, 2]
        meta_ax.axis("off")
        meta_ax.text(
            0.0,
            1.0,
            "\n".join(
                [
                    f"Epoch: {epoch:03d}",
                    f"Case: {meta['case']}",
                    f"Layer: {meta['k']}",
                    f"Depth scalar: {float(d.item()):.4f}",
                    f"Input shape: {tuple(x.shape)}",
                    f"Injection shape: {tuple(inj.shape)}",
                    f"Inj max abs: {float(inj.abs().max().item()):.4f}",
                    f"Mean SF uncertainty: {float(uncertainty_map[:, 0].mean().item()):.4f}",
                    f"Mean VD uncertainty: {float(uncertainty_map[:, 1].mean().item()):.4f}",
                    f"Mean SF confidence: {float(confidence_map[:, 0].mean().item()):.4f}",
                    f"Mean VD confidence: {float(confidence_map[:, 1].mean().item()):.4f}",
                ]
            ),
            va="top",
            ha="left",
            fontsize=11,
        )

        input_fig.suptitle(
            f"Epoch {epoch:03d} Inputs | case={meta['case']} | layer={meta['k']}",
            fontsize=14,
        )
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.savefig(path / f"epoch_{epoch:03d}_inputs.png")
        plt.close(input_fig)

        temporal_fig, temporal_axes = plt.subplots(1, 2, figsize=(14, 5))
        sf_unc_time = uncertainty_map[:, 0].mean(dim=(1, 2)).numpy()
        vd_unc_time = uncertainty_map[:, 1].mean(dim=(1, 2)).numpy()
        temporal_axes[0].plot(np.arange(len(sf_unc_time)), sf_unc_time, linewidth=2.0, color="tab:blue")
        temporal_axes[0].set_title("SF mean uncertainty by time")
        temporal_axes[0].set_xlabel("Time")
        temporal_axes[0].set_ylabel("0-1")
        temporal_axes[0].grid(alpha=0.3)

        temporal_axes[1].plot(np.arange(len(vd_unc_time)), vd_unc_time, linewidth=2.0, color="tab:orange")
        temporal_axes[1].set_title("VD mean uncertainty by time")
        temporal_axes[1].set_xlabel("Time")
        temporal_axes[1].set_ylabel("0-1")
        temporal_axes[1].grid(alpha=0.3)

        temporal_fig.suptitle(
            f"Epoch {epoch:03d} uncertainty timeline | case={meta['case']} | layer={meta['k']}",
            fontsize=14,
        )
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        plt.savefig(path / f"epoch_{epoch:03d}_uncertainty_timeline.png")
        plt.close(temporal_fig)
