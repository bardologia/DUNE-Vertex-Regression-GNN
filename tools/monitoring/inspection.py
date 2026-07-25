from __future__ import annotations

from pathlib import Path

import torch.nn as nn

from tools.reporting.markdown import MarkdownDoc, MarkdownTable


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
        self.total_params = self.count_parameters(self.model)

        for name, module in self.model.named_modules():
            if name == "":
                continue

            parameters = sum(parameter.numel() for parameter in module.parameters(recurse=False))
            if parameters == 0:
                continue

            self.rows.append((name, module.__class__.__name__, parameters))
