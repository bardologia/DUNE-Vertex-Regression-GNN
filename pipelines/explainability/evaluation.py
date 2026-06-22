from __future__ import annotations

import torch
from torch_geometric.data   import Data
from torch_geometric.loader import DataLoader as GraphDataLoader

from tools.monitoring.logger     import NullLogger
from pipelines.inference.metrics import InferenceMetrics


class EvaluationGraphTensors:
    def __init__(self, dataset, batch_size, max_events, logger):
        self.dataset     = dataset
        self.batch_size  = batch_size
        self.max_events  = max_events
        self.logger      = logger

        self.graphs      = None
        self.node_matrix = None
        self.edge_matrix = None
        self.node_slices = None
        self.edge_slices = None

    def _materialize(self):
        count = len(self.dataset)
        if self.max_events and self.max_events > 0:
            count = min(count, self.max_events)
        self.graphs = [self.dataset[index] for index in range(count)]

    def _stack(self):
        node_segments = []
        edge_segments = []
        node_slices   = []
        edge_slices   = []
        node_cursor   = 0
        edge_cursor   = 0

        for graph in self.graphs:
            node_count = int(graph.x.shape[0])
            edge_count = int(graph.edge_attr.shape[0])

            node_slices.append((node_cursor, node_cursor + node_count))
            edge_slices.append((edge_cursor, edge_cursor + edge_count))

            node_segments.append(graph.x)
            edge_segments.append(graph.edge_attr)

            node_cursor += node_count
            edge_cursor += edge_count

        self.node_matrix = torch.cat(node_segments, dim=0).clone()
        self.edge_matrix = torch.cat(edge_segments, dim=0).clone()
        self.node_slices = node_slices
        self.edge_slices = edge_slices

    def node_feature_count(self):
        return int(self.node_matrix.shape[1])

    def edge_feature_count(self):
        return int(self.edge_matrix.shape[1])

    def node_row_count(self):
        return int(self.node_matrix.shape[0])

    def edge_row_count(self):
        return int(self.edge_matrix.shape[0])

    def loader(self, node_matrix=None, edge_matrix=None):
        node_source = self.node_matrix if node_matrix is None else node_matrix
        edge_source = self.edge_matrix if edge_matrix is None else edge_matrix

        rebuilt = []
        for graph, (node_start, node_end), (edge_start, edge_end) in zip(self.graphs, self.node_slices, self.edge_slices):
            rebuilt.append(Data(x=node_source[node_start:node_end], edge_index=graph.edge_index, edge_attr=edge_source[edge_start:edge_end], y=graph.y))

        return GraphDataLoader(rebuilt, batch_size=self.batch_size, shuffle=False)

    def prepare(self):
        self._materialize()
        self._stack()

        self.logger.subsection(f"Stacked {len(self.graphs)} graphs: {self.node_row_count()} node rows x {self.node_feature_count()} features | {self.edge_row_count()} edge rows x {self.edge_feature_count()} features")
        return self


class SplitMetricEvaluator:
    def __init__(self, predictor, metric_keys):
        self.predictor   = predictor
        self.metric_keys = tuple(metric_keys)
        self.metrics     = InferenceMetrics(NullLogger())

    def evaluate(self, loader):
        predictions, targets = self.predictor.predict(loader)
        results              = self.metrics.compute(predictions, targets, stage="perturbation")
        return {key: float(results[key]) for key in self.metric_keys}
