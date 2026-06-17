from __future__ import annotations

from pathlib import Path

import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import Aggregation

from tools.reporting.markdown import MarkdownDoc, MarkdownTable


class ShapeLogger:

    DEFAULT_TYPES = (
        nn.Linear,
        nn.LayerNorm,
        nn.BatchNorm1d,
        nn.Dropout,
        nn.Identity,
        nn.ReLU,
        nn.LeakyReLU,
        nn.ELU,
        nn.GELU,
        nn.Tanh,
        MessagePassing,
        Aggregation,
    )

    def __init__(self, model, logger, include_types=DEFAULT_TYPES):
        self.model         = model
        self.include_types = include_types
        self.logger        = logger
        self.records       = []
        self.hooks         = []

    def _hook(self, name):
        def forward_hook(module, inputs, output):
            input_value = inputs[0]
            input_shape = tuple(input_value.shape) if hasattr(input_value, "shape") else str(type(input_value))

            if isinstance(output, tuple):
                output_shape = tuple(output[0].shape) if hasattr(output[0], "shape") else f"tuple[{len(output)}]"
            else:
                output_shape = tuple(output.shape) if hasattr(output, "shape") else str(type(output))

            self.records.append((name, module.__class__.__name__, input_shape, output_shape))

        return forward_hook

    def attach(self):
        self.logger.subsection("Attaching forward hooks for shape logging")

        for name, module in self.model.named_modules():
            if name == "":
                continue

            if isinstance(module, self.include_types):
                self.hooks.append(module.register_forward_hook(self._hook(name)))

        return self

    def detach(self):
        if not self.hooks:
            return

        for hook in self.hooks:
            hook.remove()

        self.hooks.clear()

    def clear(self):
        self.records.clear()

    def to_markdown(self, title: str = "Shape Log", sort_by_layer: bool = False) -> str:
        rows = list(self.records)
        if sort_by_layer:
            rows.sort(key=lambda record: record[0])

        table = MarkdownTable(["Layer", "Type", "Input shape", "Output shape"], align=["left", "left", "left", "left"])
        for name, module_type, input_shape, output_shape in rows:
            table.add_row(f"`{name}`", str(module_type), str(input_shape), str(output_shape))

        document = MarkdownDoc(title)
        document.table(table)
        document.paragraph(f"Records: {len(rows)}")
        return document.render()

    def save_markdown(self, path, title: str = "Shape Log", sort_by_layer: bool = False):
        Path(path).write_text(self.to_markdown(title=title, sort_by_layer=sort_by_layer), encoding="utf-8")
        self.logger.subsection(f"Shape log saved to {path}")


class ModelSummary:
    def __init__(self, logger, model: nn.Module):
        self.logger       = logger
        self.model        = model
        self.rows         = []
        self.total_params = 0

    def count_parameters(self, module: nn.Module) -> int:
        return sum(parameter.numel() for parameter in module.parameters())

    def to_markdown(self, title: str = "Model Summary") -> str:
        table = MarkdownTable(["Layer", "Type", "Parameters"], align=["left", "left", "right"])
        for name, module_type, parameters in self.rows:
            table.add_row(name, module_type, f"{parameters:,}")

        document = MarkdownDoc(title)
        document.table(table)
        document.paragraph(f"Total Parameters: {self.total_params:,}")
        return document.render()

    def save_markdown(self, path: str, title: str = "Model Summary"):
        Path(path).write_text(self.to_markdown(title=title), encoding="utf-8")
        self.logger.subsection(f"Model summary saved to {path}")

    def run(self):
        self.logger.section("[Model Summary]")
        self.total_params = 0

        for name, module in self.model.named_modules():
            if name == "":
                continue

            parameters         = self.count_parameters(module)
            self.total_params += parameters
            self.rows.append((name, module.__class__.__name__, parameters))
