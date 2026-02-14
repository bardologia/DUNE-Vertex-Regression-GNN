import logging
import os
import sys
from datetime import datetime
from torch import nn
from pathlib import Path
from torch_geometric.nn import GATv2Conv, SAGPooling
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D


class Logger:
    
    LOG_LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    def __init__(self, log_dir="logs", name="experiment", level="INFO", config=None):
        self.log_dir = log_dir
        self.name = name
        self.start_time = datetime.now()
        self.config = config
        if log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
        
        self.logger = logging.getLogger(name)
        
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        log_level = self.LOG_LEVELS.get(str(level).upper(), logging.INFO)
        self.logger.setLevel(log_level)
        
        file_formatter = logging.Formatter(
            '[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = logging.Formatter(
            '[%(asctime)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        
        log_filename = f'{name}_{self.start_time.strftime("%Y%m%d_%H%M%S")}.log'
        if log_dir:
            file_handler = logging.FileHandler(os.path.join(self.log_dir, log_filename), encoding='utf-8')
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(log_level)
            self.logger.addHandler(file_handler)
    
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(log_level)
        self.logger.addHandler(console_handler)
    
    def section(self, title: str):
        self.logger.info("")
        self.logger.info(f">>> {str(title).upper()}")
    
    def subsection(self, title: str):
        self.logger.info(f"  > {title}")
    
    def progress(self, current: int, total: int, prefix: str = "", suffix: str = ""):
        percentage = 100 * (current / float(total))
        self.logger.info(f"{prefix} [{current}/{total}] ({percentage:.1f}%) {suffix}")

    def debug(self, message: str):
        self.logger.debug(message)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
        
    def error(self, message: str):
        self.logger.error(message)
    
    def critical(self, message: str):
        self.logger.critical(message)
            
    def close(self):
        elapsed = datetime.now() - self.start_time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        self.logger.info(f"[End] Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


class ShapeLogger:
    def __init__(self, model, logger, include_types = (
    nn.Linear,
    nn.LayerNorm,
    nn.Dropout,
    nn.Identity,
    nn.ReLU,
    nn.LeakyReLU,
    nn.ELU,
    nn.GELU,
    nn.Tanh,
    GATv2Conv,
    SAGPooling,
)):
        self.model         = model
        self.include_types = include_types
        self.records       = []
        self.hooks         = []
        self.logger        = logger
    
    def _hook(self, name):
        def fn(module, inputs, output):
            x = inputs[0]
            in_shape  = tuple(x.shape) if hasattr(x, "shape") else str(type(x))
            out_shape = tuple(output.shape) if hasattr(output, "shape") else str(type(output))
            self.records.append((name, module.__class__.__name__, in_shape, out_shape))
        
        return fn

    def attach(self):
        self.logger.info("Hooks attached to layers for shape logging. \n")
        
        for name, module in self.model.named_modules():
            if name == "":
                continue
            if isinstance(module, self.include_types):
                self.hooks.append(module.register_forward_hook(self._hook(name)))
        
        return self

    def detach(self):
        if self.hooks == []:
            return

        self.logger.info("Hooks detached from layers. \n")

        for h in self.hooks:
            h.remove()
        
        self.hooks.clear()

    def clear(self):
        self.records.clear()
    
    def to_markdown(self, title: str = "Shape Log", sort_by_layer: bool = False) -> str:
        rows = list(self.records)
        if sort_by_layer:
            rows.sort(key=lambda r: r[0])

        def s(x):
            return str(x)

        def layer_cell(name: str) -> str:
            return f"`{name}`"  # exatamente como será impresso

        col_names = ["Layer", "Type", "Input shape", "Output shape"]
        col_data = [
            [layer_cell(r[0]) for r in rows],
            [str(r[1]) for r in rows],
            [s(r[2]) for r in rows],
            [s(r[3]) for r in rows],
        ]

        widths = []
        for header, data in zip(col_names, col_data):
            widths.append(max([len(header)] + [len(v) for v in data]) if rows else len(header))

        def fmt_row(cells):
            return "| " + " | ".join(f"{c:<{w}}" for c, w in zip(cells, widths)) + " |"

        def fmt_sep():
            return "| " + " | ".join((":" + "-" * (w - 1)) if w > 1 else "-" for w in widths) + " |"

        lines = []
        lines.append(f"# {title}\n")
        lines.append(fmt_row(col_names))
        lines.append(fmt_sep())

        for (name, typ, ins, outs), layer_txt in zip(rows, col_data[0]):
            lines.append(fmt_row([layer_txt, str(typ), s(ins), s(outs)]))

        lines.append(f"\n**Records:** {len(rows)}")
        return "\n".join(lines)

    def save_markdown(self, path, title: str = "Shape Log", sort_by_layer: bool = False):
        md = self.to_markdown(title=title, sort_by_layer=sort_by_layer)
        Path(path).write_text(md, encoding="utf-8")
        self.logger.info(f"Shape log saved to {path} \n")


class ModelSummary:
    def __init__(self, logger, model: nn.Module):
        self.model        = model
        self.rows         = []
        self.total_params = 0
        self.logger       = logger
     
    def count_params(self, module: nn.Module):
        return sum(p.numel() for p in module.parameters())

    def to_markdown(self, title="Model Summary") -> str:
        if not self.rows:
            return f"# {title}\n\nNo layers found."
        
        rows_fmt = [(name, typ, f"{params:,}") for name, typ, params in self.rows]
        
        col1 = max(len("Layer"), *(len(name) for name, _, _ in rows_fmt))
        col2 = max(len("Type"), *(len(typ) for _, typ, _ in rows_fmt))
        col3 = max(len("Parameters"), *(len(p) for _, _, p in rows_fmt))

        def line(a, b, c):
            return f"| {a:<{col1}} | {b:<{col2}} | {c:>{col3}} |"

        table = []
        table.append(line("Layer", "Type", "Parameters"))
        table.append(f"| {'-'*col1} | {'-'*col2} | {'-'*col3} |")
        for name, typ, params in rows_fmt:
            table.append(line(name, typ, params))

        total = f"{self.total_params:,}"

        md = []
        md.append(f"# {title}\n")
        md.extend(table)
        md.append(f"\n**Total Parameters:** `{total}`")
        return "\n".join(md)

    def run(self):
        self.logger.section("[Model Summary]")
        self.logger.info("Generating model architecture summary")
        self.total_params = 0

        for name, module in self.model.named_modules():
            if name == "":
                continue

            n_params = self.count_params(module)
            self.total_params += n_params
            
            self.rows.append((name, module.__class__.__name__, n_params))
    
    def save_markdown(self, path: str, title: str = "Model Summary"):
        md = self.to_markdown(title=title)
        Path(path).write_text(md, encoding="utf-8")
        self.logger.info(f"Model summary saved to {path} \n")


class DiagnosticPlots:
    @staticmethod
    def _error_vs_coordinate_plots(target_coords, errors, coord_names):
        figures = []
        for idx, (coord_name, target_coord) in enumerate(zip(coord_names, target_coords)):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for error_name, error_values in errors.items():
                sorted_indices = np.argsort(target_coord)
                sorted_targets = target_coord[sorted_indices]
                sorted_errors = error_values[sorted_indices]
                ax.scatter(sorted_targets, sorted_errors, alpha=0.3, s=10, label=error_name)
            
            ax.set_xlabel(f'Target {coord_name.upper()}')
            ax.set_ylabel('Absolute Error')
            ax.set_title(f'Error vs Target {coord_name.upper()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            figures.append((f'error_vs_target_{coord_name}', fig))
        
        return figures
    
    @staticmethod
    def _error_histograms(error_x, error_y, error_z, error_euclidean):
        figures = []
        error_histograms = [
            ('x', error_x, 'Error X'),
            ('y', error_y, 'Error Y'),
            ('z', error_z, 'Error Z'),
            ('euclidean', error_euclidean, 'Euclidean Error')
        ]
        
        for coord_name, error_values, title in error_histograms:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(error_values, bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Absolute Error')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{title} Distribution')
            ax.grid(True, alpha=0.3)
            figures.append((f'error_hist_{coord_name}', fig))
        
        return figures
    
    @staticmethod
    def _error_correlation_heatmaps(error_x, error_y, error_z):
        figures = []
        coord_pairs = [('x', 'y', error_x, error_y), ('x', 'z', error_x, error_z), ('y', 'z', error_y, error_z)]
        
        for c1, c2, e1, e2 in coord_pairs:
            fig, ax = plt.subplots(figsize=(10, 8))
            h = ax.hist2d(e1, e2, bins=50, cmap='viridis')
            plt.colorbar(h[3], ax=ax, label='Count')
            ax.set_xlabel(f'Error {c1.upper()}')
            ax.set_ylabel(f'Error {c2.upper()}')
            ax.set_title(f'Error Correlation: {c1.upper()} vs {c2.upper()}')
            ax.grid(True, alpha=0.3)
            figures.append((f'error_correlation_{c1}_{c2}', fig))
        
        return figures
    
    @staticmethod
    def _residual_plots(coord_names, pred_coords, signed_errors):
        figures = []
        for coord_name, pred_coord, signed_err in zip(coord_names, pred_coords, signed_errors):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(pred_coord, signed_err, alpha=0.3, s=10)
            ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
            ax.set_xlabel(f'Predicted {coord_name.upper()}')
            ax.set_ylabel(f'Signed Error (Pred - Target)')
            ax.set_title(f'Residual Plot: {coord_name.upper()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            figures.append((f'residual_{coord_name}', fig))
        
        return figures
    
    @staticmethod
    def _qq_plots(coord_names, signed_errors, error_euclidean):
        figures = []
        
        for coord_name, signed_err in zip(coord_names, signed_errors):
            fig, ax = plt.subplots(figsize=(10, 6))
            stats.probplot(signed_err, dist="norm", plot=ax)
            ax.set_title(f'Q-Q Plot: {coord_name.upper()} Errors')
            ax.grid(True, alpha=0.3)
            figures.append((f'qq_plot_{coord_name}', fig))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        stats.probplot(error_euclidean, dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot: Euclidean Error')
        ax.grid(True, alpha=0.3)
        figures.append((f'qq_plot_euclidean', fig))
        
        return figures
    
    @staticmethod
    def _cumulative_error_plot(error_x, error_y, error_z, error_euclidean):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for coord_name, err in zip(['X', 'Y', 'Z'], [error_x, error_y, error_z]):
            sorted_err = np.sort(err)
            cumulative = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100
            ax.plot(sorted_err, cumulative, label=f'{coord_name}', linewidth=2)
        
        sorted_euc = np.sort(error_euclidean)
        cumulative_euc = np.arange(1, len(sorted_euc) + 1) / len(sorted_euc) * 100
        ax.plot(sorted_euc, cumulative_euc, label='Euclidean', linewidth=2, linestyle='--')
        
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Cumulative Percentage (%)')
        ax.set_title('Cumulative Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def _error_vs_magnitude_plot(target_x, target_y, target_z, error_euclidean):
        target_magnitude = np.sqrt(target_x**2 + target_y**2 + target_z**2)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(target_magnitude, error_euclidean, alpha=0.3, s=10)
        ax.set_xlabel('Target Distance from Origin')
        ax.set_ylabel('Euclidean Error')
        ax.set_title('Error vs Distance from Origin')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def _coordinate_bias_plot(coord_names, signed_errors):
        bias_values = [err.mean() for err in signed_errors]
        bias_std = [err.std() for err in signed_errors]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(coord_names))
        ax.bar(x_pos, bias_values, yerr=bias_std, capsize=5, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([c.upper() for c in coord_names])
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_ylabel('Mean Signed Error')
        ax.set_title('Per-Coordinate Bias')
        ax.grid(True, alpha=0.3, axis='y')
        
        return fig
    
    @staticmethod
    def _calibration_plots(coord_names, pred_coords, target_coords):
        figures = []
        
        for coord_name, pred_coord, target_coord in zip(coord_names, pred_coords, target_coords):
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(target_coord, pred_coord, alpha=0.3, s=10)
            
            min_val = min(target_coord.min(), pred_coord.min())
            max_val = max(target_coord.max(), pred_coord.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            ax.set_xlabel(f'Target {coord_name.upper()}')
            ax.set_ylabel(f'Predicted {coord_name.upper()}')
            ax.set_title(f'Prediction Calibration: {coord_name.upper()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            figures.append((f'calibration_{coord_name}', fig))
        
        return figures
    
    @staticmethod
    def _error_boxplots(error_x, error_y, error_z, error_euclidean):
        fig, axes = plt.subplots(1, 4, figsize=(16, 6))
        box_data = [error_x, error_y, error_z, error_euclidean]
        box_labels = ['X', 'Y', 'Z', 'Euclidean']
        
        for ax, data, label in zip(axes, box_data, box_labels):
            ax.boxplot(data, labels=[label])
            ax.set_ylabel('Absolute Error')
            ax.set_title(f'{label} Error Distribution')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def _worst_case_plot(target_x, target_y, target_z, error_euclidean, k=100):
        target_magnitude = np.sqrt(target_x**2 + target_y**2 + target_z**2)
        k = min(k, len(error_euclidean))
        worst_indices = np.argsort(error_euclidean)[-k:]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(target_magnitude, error_euclidean, alpha=0.3, s=10, label='All Predictions')
        ax.scatter(target_magnitude[worst_indices], error_euclidean[worst_indices], color='red', s=30, alpha=0.7, label=f'Top {k} Worst')
        ax.set_xlabel('Target Distance from Origin')
        ax.set_ylabel('Euclidean Error')
        ax.set_title(f'Worst Case Analysis (Top {k})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def _3d_scatter_plot(pred_x, pred_y, pred_z, target_x, target_y, target_z, error_euclidean):
        fig = plt.figure(figsize=(14, 6))
        
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(target_x, target_y, target_z, c='blue', alpha=0.3, s=10, label='Target')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Target Positions')
        ax1.legend()
        
        ax2 = fig.add_subplot(122, projection='3d')
        scatter = ax2.scatter(pred_x, pred_y, pred_z, c=error_euclidean, cmap='viridis', alpha=0.5, s=10)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('Predictions (colored by error)')
        plt.colorbar(scatter, ax=ax2, label='Euclidean Error')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def _spatial_error_map(target_x, target_y, target_z, error_euclidean, n_bins=20):
        x_bins = np.linspace(target_x.min(), target_x.max(), n_bins)
        y_bins = np.linspace(target_y.min(), target_y.max(), n_bins)
        z_bins = np.linspace(target_z.min(), target_z.max(), n_bins)
        
        H_error, edges = np.histogramdd(
            np.column_stack([target_x, target_y, target_z]),
            bins=[x_bins, y_bins, z_bins],
            weights=error_euclidean
        )
        H_count, _ = np.histogramdd(
            np.column_stack([target_x, target_y, target_z]),
            bins=[x_bins, y_bins, z_bins]
        )
        
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_error_voxel = H_error / H_count
            avg_error_voxel[~np.isfinite(avg_error_voxel)] = 0
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        z_indices = [n_bins//4, n_bins//2, 3*n_bins//4, -1]
        
        for ax, z_idx in zip(axes.flat, z_indices):
            im = ax.imshow(avg_error_voxel[:, :, z_idx].T,
                          origin='lower', cmap='hot', aspect='auto')
            ax.set_title(f'Average Error at Z-slice {z_idx}')
            ax.set_xlabel('X bin')
            ax.set_ylabel('Y bin')
            plt.colorbar(im, ax=ax, label='Avg Euclidean Error')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def _batch_variance_plot(batch_errors, stage="train"):
        batch_errors_np = np.array(batch_errors)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(batch_errors_np, alpha=0.7, linewidth=1)
        ax.set_xlabel('Batch Index')
        ax.set_ylabel('Batch Error')
        ax.set_title(f'Batch Error Variance - {stage.capitalize()}')
        ax.grid(True, alpha=0.3)
        
        if len(batch_errors_np) > 10:
            window = min(50, len(batch_errors_np) // 10)
            rolling_avg = np.convolve(batch_errors_np, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(batch_errors_np)), rolling_avg,
                   color='red', linewidth=2, label=f'Rolling Avg (window={window})')
            ax.legend()
        
        return fig
    
    @staticmethod
    def _data_distribution_plots(coords_by_split):
        figures = []
        
        for coord_idx, coord_name in enumerate(['X', 'Y', 'Z']):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for split_name, coords in coords_by_split.items():
                if coords.shape[-1] > coord_idx:
                    ax.hist(coords[:, coord_idx], bins=50, alpha=0.5, label=split_name)
            
            ax.set_xlabel(f'{coord_name} Coordinate')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Data Distribution: {coord_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            figures.append((coord_name.lower(), fig))
        
        return figures
    
    @staticmethod
    def _outlier_detection_plot(pred_x, pred_y, pred_z, target_x, target_y, target_z, error_euclidean, threshold_percentile=95):
        threshold = np.percentile(error_euclidean, threshold_percentile)
        outlier_mask = error_euclidean > threshold
        
        fig = plt.figure(figsize=(14, 6))
        
        ax1 = fig.add_subplot(121, projection='3d')
        
        ax1.scatter(target_x[~outlier_mask], target_y[~outlier_mask], target_z[~outlier_mask], c='blue', alpha=0.3, s=10, marker='o', label='Target (Normal)')
        ax1.scatter(target_x[outlier_mask],  target_y[outlier_mask],  target_z[outlier_mask],  c='red',  alpha=0.7, s=30, marker='o', label=f'Target (Outlier)')
        
        ax1.scatter(pred_x[~outlier_mask], pred_y[~outlier_mask], pred_z[~outlier_mask], c='lightgreen', alpha=0.3, s=10, marker='^', label='Pred (Normal)')
        ax1.scatter(pred_x[outlier_mask],  pred_y[outlier_mask],  pred_z[outlier_mask],  c='orange',     alpha=0.7, s=30, marker='^', label=f'Pred (Outlier)')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Predictions vs Targets with Outliers')
        ax1.legend()
        
        ax2 = fig.add_subplot(122)
        ax2.hist(error_euclidean, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(threshold, color='r', linestyle='--', linewidth=2, label=f'{threshold_percentile}th percentile: {threshold:.3f}')
        ax2.set_xlabel('Euclidean Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution with Outlier Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, threshold, outlier_mask


class Tracker:
    def __init__(self, writer):
        self.writer = writer
    
    def log_scalar(self, name: str, value, step: int):
        val = value.item() if hasattr(value, 'item') else value
        self.writer.add_scalar(name, val, step)
    
    def log_dict(self, prefix: str, data_dict: dict, step: int, add_comparison: bool = True):
        comparison_dict = {}
        for key, value in data_dict.items():
            val = value.item() if hasattr(value, 'item') else value
            self.writer.add_scalar(f'{prefix}/{key}', val, step)
            comparison_dict[key] = val
        
        if add_comparison and len(comparison_dict) > 1:
            self.writer.add_scalars(f'{prefix}/comparison', comparison_dict, step)
    
    def log_comparison(self, prefix: str, dict1: dict, dict2: dict, step: int):

        self.log_dict(f'{prefix}/before', dict1, step, add_comparison=True)
        self.log_dict(f'{prefix}/after', dict2, step, add_comparison=True)
        
        if set(dict1.keys()) == set(dict2.keys()):
            delta_dict = {}
            for key in dict1.keys():
                v1 = dict1[key].item() if hasattr(dict1[key], 'item') else dict1[key]
                v2 = dict2[key].item() if hasattr(dict2[key], 'item') else dict2[key]
                delta_dict[key] = v2 - v1
            
            if delta_dict:
                if self.writer:
                    self.writer.add_scalars(f'{prefix}/delta', delta_dict, step)
    
    def log_grouped_dict(self, prefix: str, data_dict: dict, step: int, group_patterns: list = None):
        for key, value in data_dict.items():
            val = value.item() if hasattr(value, 'item') else value
            if self.writer:
                self.writer.add_scalar(f'{prefix}/{key}', val, step)
        
        if group_patterns:
            for pattern in group_patterns:
                grouped = {}
                for key, value in data_dict.items():
                    if pattern in key:
                        val = value.item() if hasattr(value, 'item') else value
                        grouped[key] = val
                
                if len(grouped) > 1:
                    if self.writer:
                        self.writer.add_scalars(f'{prefix}/{pattern}_group', grouped, step)
    
    def log_split_dict(self, prefix: str, data_dict: dict, step: int, prefixes: tuple):
        for key, value in data_dict.items():
            val = value.item() if hasattr(value, 'item') else value
            self.writer.add_scalar(f'{prefix}/{key}', val, step)
        
        for group_prefix in prefixes:
            grouped = {}
            for key, value in data_dict.items():
                if key.startswith(group_prefix):
                    clean_key = key.replace(f'{group_prefix}_', '')
                    val = value.item() if hasattr(value, 'item') else value
                    grouped[clean_key] = val
            
            if grouped:
                self.writer.add_scalars(f'{prefix}/{group_prefix}', grouped, step)
    
    def log_optimizer(self, optimizer, step: int):
        state_dict = {}
        for i, param_group in enumerate(optimizer.param_groups):
            component_name = param_group.get('name', f'group_{i}')
        
            lr = param_group['lr']
            self.writer.add_scalar(f'optimizer/lr_{component_name}', lr, step)
            state_dict[f'lr_{component_name}'] = lr
            
            for key in ['momentum', 'weight_decay', 'eps']:
                if key in param_group:
                    val = param_group[key]
                    self.writer.add_scalar(f'optimizer/{key}_{component_name}', val, step)
                    state_dict[f'{key}_{component_name}'] = val
        
        self.writer.add_scalars(f'optimizer/comparison', state_dict, step)
    
    def log_gradients(self, model, step: int, max_grad_norm: float = None):
        total_norm       = 0.0
        total_grad_sum   = 0.0
        total_grad_count = 0
        total_zero_grads = 0
        
        layer_norms       = {}
        layer_stats       = {}
        grad_param_ratios = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad       = param.grad.detach()
                param_data = param.detach()
                
                param_norm  = grad.norm(2).item()
                total_norm += param_norm ** 2
            
                grad_flat     = grad.flatten()
                grad_mean     = grad_flat.mean().item()
                grad_std      = grad_flat.std().item()
                grad_min      = grad_flat.min().item()
                grad_max      = grad_flat.max().item()
                grad_abs_mean = grad_flat.abs().mean().item()
                
                total_grad_sum   += grad_flat.sum().item()
                total_grad_count += grad_flat.numel()
                
                zero_grads        = (grad_flat.abs() < 1e-8).sum().item()
                total_zero_grads += zero_grads
                zero_percent      = 100.0 * zero_grads / grad_flat.numel()
                
                param_norm_val = param_data.norm(2).item()
                if param_norm_val > 1e-8:
                    grad_param_ratio = param_norm / (param_norm_val + 1e-8)
                else:
                    grad_param_ratio = 0.0
                
                layer_norms[name] = param_norm
                layer_stats[name] = {
                    'mean': grad_mean,
                    'std': grad_std,
                    'min': grad_min,
                    'max': grad_max,
                    'abs_mean': grad_abs_mean,
                    'zero_percent': zero_percent
                }
                grad_param_ratios[name] = grad_param_ratio
             
                self.writer.add_scalar(f'gradients/layer_norm/{name}',         param_norm, step)
                self.writer.add_scalar(f'gradients/layer_abs_mean/{name}',     grad_abs_mean, step)
                self.writer.add_scalar(f'gradients/layer_zero_percent/{name}', zero_percent, step)
                self.writer.add_scalar(f'gradients/grad_param_ratio/{name}',   grad_param_ratio, step)
                self.writer.add_histogram(f'gradients/histogram/{name}',       grad_flat, step)
        
        total_norm = total_norm ** 0.5
        
        clip_ratio     = total_norm / max_grad_norm
        is_clipped     = total_norm > max_grad_norm
        effective_norm = min(total_norm, max_grad_norm)
        
        self.writer.add_scalar(f'gradients/clip_ratio',       clip_ratio, step)
        self.writer.add_scalar(f'gradients/is_clipped',       float(is_clipped), step)
        self.writer.add_scalar(f'gradients/effective_norm',   effective_norm, step)
        self.writer.add_scalar(f'gradients/clip_coefficient', min(1.0, max_grad_norm / (total_norm + 1e-6)), step)
        
        avg_grad = total_grad_sum / max(1, total_grad_count)
        zero_grad_percent = 100.0 * total_zero_grads / max(1, total_grad_count)
        
        self.writer.add_scalar(f'gradients/total_norm',        total_norm, step)
        self.writer.add_scalar(f'gradients/avg_gradient',      avg_grad, step)
        self.writer.add_scalar(f'gradients/zero_grad_percent', zero_grad_percent, step)
    
    def log_error_vs_coordinates(self, preds, targets, step: int, stage: str = "validation"):
        diff = preds - targets
        abs_diff = torch.abs(diff)
        
        error_x = abs_diff[:, 0].cpu().numpy()
        error_y = abs_diff[:, 1].cpu().numpy()
        error_z = abs_diff[:, 2].cpu().numpy()
        error_euclidean = torch.sqrt((diff ** 2).sum(dim=1)).cpu().numpy()
        
        signed_error_x = diff[:, 0].cpu().numpy()
        signed_error_y = diff[:, 1].cpu().numpy()
        signed_error_z = diff[:, 2].cpu().numpy()
        
        pred_x = preds[:, 0].cpu().numpy()
        pred_y = preds[:, 1].cpu().numpy()
        pred_z = preds[:, 2].cpu().numpy()
        
        target_x = targets[:, 0].cpu().numpy()
        target_y = targets[:, 1].cpu().numpy()
        target_z = targets[:, 2].cpu().numpy()
        
        coord_names = ['x', 'y', 'z']
        target_coords = [target_x, target_y, target_z]
        pred_coords = [pred_x, pred_y, pred_z]
        signed_errors = [signed_error_x, signed_error_y, signed_error_z]
        
        errors = {
            'Error X': error_x,
            'Error Y': error_y,
            'Error Z': error_z,
            'Euclidean': error_euclidean
        }
        
        figures = DiagnosticPlots._error_vs_coordinate_plots(target_coords, errors, coord_names)
        for name, fig in figures:
            self.writer.add_figure(f'error_analysis/{stage}_{name}', fig, step)
            plt.close(fig)
        
        figures = DiagnosticPlots._error_histograms(error_x, error_y, error_z, error_euclidean)
        for name, fig in figures:
            self.writer.add_figure(f'error_analysis/{stage}_{name}', fig, step)
            plt.close(fig)
        
        figures = DiagnosticPlots._error_correlation_heatmaps(error_x, error_y, error_z)
        for name, fig in figures:
            self.writer.add_figure(f'error_analysis/{stage}_{name}', fig, step)
            plt.close(fig)
        
        figures = DiagnosticPlots._residual_plots(coord_names, pred_coords, signed_errors)
        for name, fig in figures:
            self.writer.add_figure(f'error_analysis/{stage}_{name}', fig, step)
            plt.close(fig)
        
        figures = DiagnosticPlots._qq_plots(coord_names, signed_errors, error_euclidean)
        for name, fig in figures:
            self.writer.add_figure(f'error_analysis/{stage}_{name}', fig, step)
            plt.close(fig)
        
        fig = DiagnosticPlots._cumulative_error_plot(error_x, error_y, error_z, error_euclidean)
        self.writer.add_figure(f'error_analysis/{stage}_cumulative_error', fig, step)
        plt.close(fig)
        
        fig = DiagnosticPlots._error_vs_magnitude_plot(target_x, target_y, target_z, error_euclidean)
        self.writer.add_figure(f'error_analysis/{stage}_error_vs_magnitude', fig, step)
        plt.close(fig)
        
        fig = DiagnosticPlots._coordinate_bias_plot(coord_names, signed_errors)
        self.writer.add_figure(f'error_analysis/{stage}_coordinate_bias', fig, step)
        plt.close(fig)
        
        figures = DiagnosticPlots._calibration_plots(coord_names, pred_coords, target_coords)
        for name, fig in figures:
            self.writer.add_figure(f'error_analysis/{stage}_{name}', fig, step)
            plt.close(fig)
        
        fig = DiagnosticPlots._error_boxplots(error_x, error_y, error_z, error_euclidean)
        self.writer.add_figure(f'error_analysis/{stage}_error_boxplots', fig, step)
        plt.close(fig)
        
        fig = DiagnosticPlots._worst_case_plot(target_x, target_y, target_z, error_euclidean)
        self.writer.add_figure(f'error_analysis/{stage}_worst_cases', fig, step)
        plt.close(fig)

        fig = DiagnosticPlots._3d_scatter_plot(pred_x, pred_y, pred_z, target_x, target_y, target_z, error_euclidean)
        self.writer.add_figure(f'error_analysis/{stage}_3d_scatter', fig, step)
        plt.close(fig)
        
        fig = DiagnosticPlots._spatial_error_map(target_x, target_y, target_z, error_euclidean)
        self.writer.add_figure(f'error_analysis/{stage}_spatial_error_map', fig, step)
        plt.close(fig)
        
        # 14. Error statistics as scalars
        self.writer.add_scalar(f'error_stats/{stage}/mean_error_x', error_x.mean(), step)
        self.writer.add_scalar(f'error_stats/{stage}/mean_error_y', error_y.mean(), step)
        self.writer.add_scalar(f'error_stats/{stage}/mean_error_z', error_z.mean(), step)
        self.writer.add_scalar(f'error_stats/{stage}/mean_error_euclidean', error_euclidean.mean(), step)
        
        self.writer.add_scalar(f'error_stats/{stage}/median_error_x', np.median(error_x), step)
        self.writer.add_scalar(f'error_stats/{stage}/median_error_y', np.median(error_y), step)
        self.writer.add_scalar(f'error_stats/{stage}/median_error_z', np.median(error_z), step)
        self.writer.add_scalar(f'error_stats/{stage}/median_error_euclidean', np.median(error_euclidean), step)
        
        self.writer.add_scalar(f'error_stats/{stage}/std_error_x', error_x.std(), step)
        self.writer.add_scalar(f'error_stats/{stage}/std_error_y', error_y.std(), step)
        self.writer.add_scalar(f'error_stats/{stage}/std_error_z', error_z.std(), step)
        self.writer.add_scalar(f'error_stats/{stage}/std_error_euclidean', error_euclidean.std(), step)
        
        self.writer.add_scalar(f'error_stats/{stage}/max_error_euclidean', error_euclidean.max(), step)
        self.writer.add_scalar(f'error_stats/{stage}/p95_error_euclidean', np.percentile(error_euclidean, 95), step)
        self.writer.add_scalar(f'error_stats/{stage}/p99_error_euclidean', np.percentile(error_euclidean, 99), step)
        
        self.writer.add_scalar(f'error_stats/{stage}/bias_x', signed_error_x.mean(), step)
        self.writer.add_scalar(f'error_stats/{stage}/bias_y', signed_error_y.mean(), step)
        self.writer.add_scalar(f'error_stats/{stage}/bias_z', signed_error_z.mean(), step)
    
    def log_activations(self, model, step: int):
        activation_stats = {}
        
        def get_activation_hook(name):
            def hook(module, input, output):
                if hasattr(output, 'detach'):
                    act = output.detach()
                    act_flat = act.flatten()
                    
                    activation_stats[name] = {
                        'mean': act_flat.mean().item(),
                        'std': act_flat.std().item(),
                        'min': act_flat.min().item(),
                        'max': act_flat.max().item(),
                        'abs_mean': act_flat.abs().mean().item(),
                    }
                    
                    self.writer.add_histogram(f'activations/{name}', act_flat.cpu(), step)
                    self.writer.add_scalar(f'activations/mean/{name}', activation_stats[name]['mean'], step)
                    self.writer.add_scalar(f'activations/std/{name}', activation_stats[name]['std'], step)
                    self.writer.add_scalar(f'activations/abs_mean/{name}', activation_stats[name]['abs_mean'], step)
            
            return hook
        
        hooks = []
        for name, module in model.named_modules():
            if name and len(list(module.children())) == 0:  
                hooks.append(module.register_forward_hook(get_activation_hook(name)))
        
        return hooks 
    
    def log_weights(self, model, step: int):
        for name, param in model.named_parameters():
            if param.requires_grad and 'weight' in name:
                weight_flat = param.detach().flatten().cpu()
                
                self.writer.add_histogram(f'weights/{name}', weight_flat, step)
                
                self.writer.add_scalar(f'weights/mean/{name}', weight_flat.mean().item(), step)
                self.writer.add_scalar(f'weights/std/{name}', weight_flat.std().item(), step)
                self.writer.add_scalar(f'weights/abs_mean/{name}', weight_flat.abs().mean().item(), step)
                self.writer.add_scalar(f'weights/norm/{name}', weight_flat.norm(2).item(), step)
              
                dead_weights = (weight_flat.abs() < 1e-8).sum().item()
                dead_percent = 100.0 * dead_weights / weight_flat.numel()
                self.writer.add_scalar(f'weights/dead_percent/{name}', dead_percent, step)
    
    def log_batch_statistics(self, batch_errors: list, step: int, stage: str = "train"):
        batch_errors_np = np.array(batch_errors)
        
        fig = DiagnosticPlots._batch_variance_plot(batch_errors, stage)
        self.writer.add_figure(f'batch_stats/{stage}_error_variance', fig, step)
        plt.close(fig)
        
        self.writer.add_scalar(f'batch_stats/{stage}/mean', batch_errors_np.mean(), step)
        self.writer.add_scalar(f'batch_stats/{stage}/std', batch_errors_np.std(), step)
        self.writer.add_scalar(f'batch_stats/{stage}/min', batch_errors_np.min(), step)
        self.writer.add_scalar(f'batch_stats/{stage}/max', batch_errors_np.max(), step)
        self.writer.add_scalar(f'batch_stats/{stage}/range', batch_errors_np.max() - batch_errors_np.min(), step)
    
    def log_data_distribution(self, train_data, val_data, test_data, step: int):
        datasets = {'train': train_data, 'val': val_data, 'test': test_data}
        
        coords_by_split = {}
        for name, data in datasets.items():
            if data is not None:
                if hasattr(data, 'pos'):
                    coords = data.pos.cpu().numpy()
                elif isinstance(data, torch.Tensor):
                    coords = data.cpu().numpy()
                else:
                    coords = np.array(data)
                coords_by_split[name] = coords

        figures = DiagnosticPlots._data_distribution_plots(coords_by_split)
        for coord_name, fig in figures:
            self.writer.add_figure(f'data_distribution/{coord_name}', fig, step)
            plt.close(fig)
    
    def log_outlier_detection(self, preds, targets, step: int, stage: str = "validation", threshold_percentile: float = 95):
        diff = preds - targets
        error_euclidean = torch.sqrt((diff ** 2).sum(dim=1)).cpu().numpy()
        
        pred_np = preds.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        fig, threshold, outlier_mask = DiagnosticPlots._outlier_detection_plot(
            pred_np[:, 0],   pred_np[:, 1],   pred_np[:, 2],
            target_np[:, 0], target_np[:, 1], target_np[:, 2],
            error_euclidean, threshold_percentile
        )
        
        self.writer.add_figure(f'outlier_detection/{stage}_outliers', fig, step)
        plt.close(fig)
        
        n_outliers = outlier_mask.sum()
        outlier_percent = 100.0 * n_outliers / len(error_euclidean)
        self.writer.add_scalar(f'outlier_detection/{stage}/count', n_outliers, step)
        self.writer.add_scalar(f'outlier_detection/{stage}/percent', outlier_percent, step)
        self.writer.add_scalar(f'outlier_detection/{stage}/threshold', threshold, step)
        
    def close(self):
        self.writer.close()