from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data   import Data
from torch_geometric.loader import DataLoader as GraphDataLoader


class PerturbationImportance:
    DOMAIN_OFFSET  = {"node": 0, "edge": 100_000}
    COLUMN_STRIDE  = 131
    REPEAT_STRIDE  = 7919

    def __init__(self, engine, evaluator, baseline, metric_keys, primary_metric, seed, logger):
        self.engine         = engine
        self.evaluator      = evaluator
        self.baseline       = baseline
        self.metric_keys    = tuple(metric_keys)
        self.primary_metric = primary_metric
        self.seed           = seed
        self.logger         = logger

    def _repeats(self):
        raise NotImplementedError

    def _base_matrix(self, domain):
        return self.engine.node_matrix if domain == "node" else self.engine.edge_matrix

    def _row_count(self, domain):
        return self.engine.node_row_count() if domain == "node" else self.engine.edge_row_count()

    def _loader(self, domain, matrix):
        if domain == "node":
            return self.engine.loader(node_matrix=matrix)
        return self.engine.loader(edge_matrix=matrix)

    def _perturb(self, domain, columns, repeat_index):
        raise NotImplementedError

    def _aggregate(self, samples):
        deltas = {key: float(np.mean([sample[key] for sample in samples])) for key in self.metric_keys}

        primary_values = np.array([sample[self.primary_metric] for sample in samples], dtype=np.float64)
        primary_std    = float(primary_values.std(ddof=0))

        baseline_primary = self.baseline[self.primary_metric]
        pct_increase     = float(100.0 * deltas[self.primary_metric] / baseline_primary) if baseline_primary != 0.0 else float("nan")
        return deltas, primary_std, pct_increase

    def _evaluate_item(self, domain, columns, progress, task_id):
        samples = []
        for repeat_index in range(self._repeats()):
            matrix  = self._perturb(domain, columns, repeat_index)
            metrics = self.evaluator.evaluate(self._loader(domain, matrix))
            samples.append({key: metrics[key] - self.baseline[key] for key in self.metric_keys})
            progress.advance(task_id)

        deltas, primary_std, pct_increase = self._aggregate(samples)
        return {"deltas": deltas, "primary_std": primary_std, "pct_increase": pct_increase}

    def _collection(self, domain, items, progress, task_id):
        records = []
        for label, group, columns in items:
            outcome = self._evaluate_item(domain, columns, progress, task_id)
            records.append({"label": label, "group": group, "columns": list(columns), **outcome})
        return records

    def _feature_items(self, columns_layout):
        return [(name, group, [index]) for index, name, group in columns_layout]

    def _group_items(self, groups):
        return [(group, group, indices) for group, indices in groups.items()]

    def _total_steps(self, node_columns, node_groups, edge_columns, edge_groups):
        node_items = len(node_columns) + len(node_groups)
        edge_items = len(edge_columns) + len(edge_groups)
        return (node_items + edge_items) * self._repeats()

    def run(self, node_columns, node_groups, edge_columns, edge_groups):
        self.logger.section(f"[{self.label()}]")
        results = {}

        with self.logger.track() as progress:
            task_id = progress.add_task(self.label(), total=self._total_steps(node_columns, node_groups, edge_columns, edge_groups))

            results["node_features"] = self._collection("node", self._feature_items(node_columns), progress, task_id)
            results["node_groups"]   = self._collection("node", self._group_items(node_groups),    progress, task_id)
            results["edge_features"] = self._collection("edge", self._feature_items(edge_columns), progress, task_id)
            results["edge_groups"]   = self._collection("edge", self._group_items(edge_groups),    progress, task_id)

        return results


class PermutationImportance(PerturbationImportance):
    def __init__(self, engine, evaluator, baseline, metric_keys, primary_metric, seed, repeats, logger):
        super().__init__(engine, evaluator, baseline, metric_keys, primary_metric, seed, logger)
        self.repeats = repeats

    def label(self):
        return "Permutation Importance"

    def _repeats(self):
        return self.repeats

    def _perturb(self, domain, columns, repeat_index):
        identifier  = self.seed + self.DOMAIN_OFFSET[domain] + columns[0] * self.COLUMN_STRIDE + repeat_index * self.REPEAT_STRIDE
        generator   = np.random.default_rng(identifier)
        permutation = torch.from_numpy(generator.permutation(self._row_count(domain)))

        matrix              = self._base_matrix(domain).clone()
        column_index        = torch.tensor(columns, dtype=torch.long)
        matrix[:, column_index] = matrix[permutation][:, column_index]
        return matrix


class OcclusionImportance(PerturbationImportance):
    def label(self):
        return "Occlusion Importance"

    def _repeats(self):
        return 1

    def _perturb(self, domain, columns, repeat_index):
        base                = self._base_matrix(domain)
        matrix              = base.clone()
        column_index        = torch.tensor(columns, dtype=torch.long)
        matrix[:, column_index] = base[:, column_index].mean(dim=0, keepdim=True)
        return matrix


class GradientSaliency:
    def __init__(self, model, device, engine, logger):
        self.model  = model
        self.device = device
        self.engine = engine
        self.logger = logger

    def _zero_accumulators(self):
        node_features = self.engine.node_feature_count()
        edge_features = self.engine.edge_feature_count()
        return {
            "node_gradient" : torch.zeros(node_features, dtype=torch.float64),
            "node_product"  : torch.zeros(node_features, dtype=torch.float64),
            "edge_gradient" : torch.zeros(edge_features, dtype=torch.float64),
            "edge_product"  : torch.zeros(edge_features, dtype=torch.float64),
        }

    def _accumulate_batch(self, data, accumulators, counters):
        data = data.to(self.device)
        data.x.requires_grad_(True)
        data.edge_attr.requires_grad_(True)

        predictions = self.model(data)
        loss        = ((predictions - data.y) ** 2).sum(dim=1).mean()

        self.model.zero_grad(set_to_none=True)
        loss.backward()

        node_gradient = data.x.grad.detach().abs()
        node_product  = (data.x.detach() * data.x.grad.detach()).abs()
        accumulators["node_gradient"] += node_gradient.sum(dim=0).double().cpu()
        accumulators["node_product"]  += node_product.sum(dim=0).double().cpu()
        counters["node"]              += int(data.x.shape[0])

        if data.edge_attr.grad is not None:
            edge_gradient = data.edge_attr.grad.detach().abs()
            edge_product  = (data.edge_attr.detach() * data.edge_attr.grad.detach()).abs()
            accumulators["edge_gradient"] += edge_gradient.sum(dim=0).double().cpu()
            accumulators["edge_product"]  += edge_product.sum(dim=0).double().cpu()
            counters["edge"]              += int(data.edge_attr.shape[0])

    def _means(self, accumulators, counters):
        node_divisor = max(counters["node"], 1)
        edge_divisor = max(counters["edge"], 1)
        return {
            "node_saliency"         : (accumulators["node_gradient"] / node_divisor).numpy(),
            "node_input_times_grad" : (accumulators["node_product"]  / node_divisor).numpy(),
            "edge_saliency"         : (accumulators["edge_gradient"] / edge_divisor).numpy(),
            "edge_input_times_grad" : (accumulators["edge_product"]  / edge_divisor).numpy(),
        }

    def run(self):
        self.logger.section("[Gradient Saliency]")

        loader       = self.engine.loader()
        accumulators = self._zero_accumulators()
        counters     = {"node": 0, "edge": 0}

        self.model.eval()
        with self.logger.track() as progress:
            task_id = progress.add_task("Gradient saliency", total=len(loader))
            for data in loader:
                self._accumulate_batch(data, accumulators, counters)
                progress.advance(task_id)

        return self._means(accumulators, counters)


class ExpectedGradients:
    COORDINATES = ("x", "y", "z")

    def __init__(self, model, device, engine, samples, events, seed, logger):
        self.model      = model
        self.device     = device
        self.engine     = engine
        self.samples    = samples
        self.events     = events
        self.seed       = seed
        self.logger     = logger

        self.coordinate_count = len(self.COORDINATES)

        self.subset_graphs      = None
        self.subset_node_matrix = None
        self.subset_edge_matrix = None
        self.subset_node_slices = None
        self.subset_edge_slices = None
        self.node_signed        = None
        self.edge_signed        = None
        self.real_output        = None
        self.baseline_output    = None

    def _select_subset(self):
        count          = len(self.engine.graphs)
        if self.events and self.events > 0:
            count = min(count, self.events)
        self.subset_graphs = self.engine.graphs[:count]

        node_segments = []
        edge_segments = []
        node_slices   = []
        edge_slices   = []
        node_cursor   = 0
        edge_cursor   = 0

        for graph in self.subset_graphs:
            node_count = int(graph.x.shape[0])
            edge_count = int(graph.edge_attr.shape[0])

            node_slices.append((node_cursor, node_cursor + node_count))
            edge_slices.append((edge_cursor, edge_cursor + edge_count))

            node_segments.append(graph.x)
            edge_segments.append(graph.edge_attr)

            node_cursor += node_count
            edge_cursor += edge_count

        self.subset_node_matrix = torch.cat(node_segments, dim=0)
        self.subset_edge_matrix = torch.cat(edge_segments, dim=0)
        self.subset_node_slices = node_slices
        self.subset_edge_slices = edge_slices

        self.node_signed     = torch.zeros(self.subset_node_matrix.shape[0], self.subset_node_matrix.shape[1], self.coordinate_count, dtype=torch.float64)
        self.edge_signed     = torch.zeros(self.subset_edge_matrix.shape[0], self.subset_edge_matrix.shape[1], self.coordinate_count, dtype=torch.float64)
        self.real_output     = torch.zeros(self.coordinate_count, dtype=torch.float64)
        self.baseline_output = torch.zeros(self.coordinate_count, dtype=torch.float64)

    def _alpha_vector(self, slices, count, alphas):
        vector = torch.empty(count, 1, dtype=torch.float32)
        for index, (start, end) in enumerate(slices):
            vector[start:end] = float(alphas[index])
        return vector

    def _data_list(self, node_matrix, edge_matrix):
        rebuilt = []
        for graph, (node_start, node_end), (edge_start, edge_end) in zip(self.subset_graphs, self.subset_node_slices, self.subset_edge_slices):
            rebuilt.append(Data(x=node_matrix[node_start:node_end], edge_index=graph.edge_index, edge_attr=edge_matrix[edge_start:edge_end], y=graph.y))
        return rebuilt

    def _sample_inputs(self, generator):
        node_baseline_index = torch.from_numpy(generator.integers(0, self.engine.node_row_count(), size=self.subset_node_matrix.shape[0]))
        edge_baseline_index = torch.from_numpy(generator.integers(0, self.engine.edge_row_count(), size=self.subset_edge_matrix.shape[0]))

        node_baseline = self.engine.node_matrix[node_baseline_index]
        edge_baseline = self.engine.edge_matrix[edge_baseline_index]

        alphas      = generator.uniform(0.0, 1.0, size=len(self.subset_graphs))
        node_alpha  = self._alpha_vector(self.subset_node_slices, self.subset_node_matrix.shape[0], alphas)
        edge_alpha  = self._alpha_vector(self.subset_edge_slices, self.subset_edge_matrix.shape[0], alphas)

        node_delta  = self.subset_node_matrix - node_baseline
        edge_delta  = self.subset_edge_matrix - edge_baseline

        node_interpolated = node_baseline + node_alpha * node_delta
        edge_interpolated = edge_baseline + edge_alpha * edge_delta
        return node_interpolated, edge_interpolated, node_delta, edge_delta, node_baseline, edge_baseline

    def _accumulate_path(self, node_interpolated, edge_interpolated, node_delta, edge_delta):
        loader      = GraphDataLoader(self._data_list(node_interpolated, edge_interpolated), batch_size=self.engine.batch_size, shuffle=False)
        node_offset = 0
        edge_offset = 0

        for data in loader:
            data = data.to(self.device)
            data.x.requires_grad_(True)
            data.edge_attr.requires_grad_(True)

            predictions = self.model(data)
            node_rows   = int(data.x.shape[0])
            edge_rows   = int(data.edge_attr.shape[0])

            node_delta_slice = node_delta[node_offset:node_offset + node_rows].to(self.device)
            edge_delta_slice = edge_delta[edge_offset:edge_offset + edge_rows].to(self.device)

            for coordinate in range(self.coordinate_count):
                data.x.grad         = None
                data.edge_attr.grad = None
                predictions[:, coordinate].sum().backward(retain_graph=coordinate < self.coordinate_count - 1)

                self.node_signed[node_offset:node_offset + node_rows, :, coordinate] += (node_delta_slice * data.x.grad).detach().double().cpu()
                if data.edge_attr.grad is not None:
                    self.edge_signed[edge_offset:edge_offset + edge_rows, :, coordinate] += (edge_delta_slice * data.edge_attr.grad).detach().double().cpu()

            node_offset += node_rows
            edge_offset += edge_rows

    def _coordinate_output(self, node_matrix, edge_matrix):
        loader = GraphDataLoader(self._data_list(node_matrix, edge_matrix), batch_size=self.engine.batch_size, shuffle=False)
        totals = torch.zeros(self.coordinate_count, dtype=torch.float64)

        with torch.no_grad():
            for data in loader:
                data         = data.to(self.device)
                predictions  = self.model(data)
                totals      += predictions.sum(dim=0).double().cpu()
        return totals

    def _completeness(self):
        attributed = (self.node_signed.sum(dim=(0, 1)) + self.edge_signed.sum(dim=(0, 1))) / self.samples
        reference  = self.real_output - self.baseline_output

        ratios = {}
        for index, name in enumerate(self.COORDINATES):
            ratios[name] = float(attributed[index] / reference[index]) if reference[index] != 0.0 else float("nan")
        return ratios

    def _means(self):
        node_phi = (self.node_signed / self.samples).numpy()
        edge_phi = (self.edge_signed / self.samples).numpy()

        return {
            "coordinates"  : self.COORDINATES,
            "node_overall" : np.abs(node_phi).sum(axis=2).mean(axis=0),
            "edge_overall" : np.abs(edge_phi).sum(axis=2).mean(axis=0),
            "node_per_axis": np.abs(node_phi).mean(axis=0).T,
            "edge_per_axis": np.abs(edge_phi).mean(axis=0).T,
            "completeness" : self._completeness(),
            "events"       : len(self.subset_graphs),
            "samples"      : self.samples,
        }

    def run(self):
        self.logger.section("[Expected Gradients]")
        self._select_subset()

        self.model.eval()
        self.real_output = self._coordinate_output(self.subset_node_matrix, self.subset_edge_matrix)

        with self.logger.track() as progress:
            task_id = progress.add_task("Expected gradients", total=self.samples)
            for sample_index in range(self.samples):
                generator                                                                  = np.random.default_rng(self.seed + sample_index)
                node_interpolated, edge_interpolated, node_delta, edge_delta, node_baseline, edge_baseline = self._sample_inputs(generator)

                self._accumulate_path(node_interpolated, edge_interpolated, node_delta, edge_delta)
                self.baseline_output += self._coordinate_output(node_baseline, edge_baseline) / self.samples
                progress.advance(task_id)

        return self._means()


class KernelShapImportance:
    COORDINATES = ("x", "y", "z")
    CHUNK_SIZE  = 64

    def __init__(self, model, device, engine, samples, events, seed, logger):
        self.model      = model
        self.device     = device
        self.engine     = engine
        self.samples    = samples
        self.events     = events
        self.seed       = seed
        self.logger     = logger

        self.coordinate_count = len(self.COORDINATES)
        self.node_features    = engine.node_feature_count()
        self.edge_features    = engine.edge_feature_count()
        self.total_features   = self.node_features + self.edge_features

        self.subset_graphs = None
        self.node_mean     = None
        self.edge_mean     = None
        self.node_sum      = None
        self.edge_sum      = None
        self.node_overall  = None
        self.edge_overall  = None
        self.completeness  = None

    def _select_subset(self):
        count          = len(self.engine.graphs)
        if self.events and self.events > 0:
            count = min(count, self.events)
        self.subset_graphs = self.engine.graphs[:count]

        self.node_mean = self.engine.node_matrix.mean(dim=0)
        self.edge_mean = self.engine.edge_matrix.mean(dim=0)

        self.node_sum     = np.zeros((self.coordinate_count, self.node_features), dtype=np.float64)
        self.edge_sum     = np.zeros((self.coordinate_count, self.edge_features), dtype=np.float64)
        self.node_overall = np.zeros(self.node_features, dtype=np.float64)
        self.edge_overall = np.zeros(self.edge_features, dtype=np.float64)
        self.completeness = {name: 0.0 for name in self.COORDINATES}

    def _masked_outputs(self, graph, gates):
        outputs = np.zeros((gates.shape[0], self.coordinate_count), dtype=np.float64)

        for start in range(0, gates.shape[0], self.CHUNK_SIZE):
            block     = gates[start:start + self.CHUNK_SIZE]
            data_list = []
            for row in block:
                node_gate = torch.from_numpy(row[: self.node_features].astype(np.float32))
                edge_gate = torch.from_numpy(row[self.node_features:].astype(np.float32))

                node = graph.x * node_gate + self.node_mean * (1.0 - node_gate)
                edge = graph.edge_attr * edge_gate + self.edge_mean * (1.0 - edge_gate)
                data_list.append(Data(x=node, edge_index=graph.edge_index, edge_attr=edge, y=graph.y))

            batch = next(iter(GraphDataLoader(data_list, batch_size=len(data_list), shuffle=False))).to(self.device)
            with torch.no_grad():
                predictions = self.model(batch)
            outputs[start:start + len(data_list)] = predictions.double().cpu().numpy()

        return outputs

    def _per_coordinate(self, shap_values):
        values = np.squeeze(np.array(shap_values))
        if values.ndim == 1:
            values = values[None, :]
        if values.shape == (self.coordinate_count, self.total_features):
            return values
        return values.T

    def _accumulate_event(self, explainer, graph, instance):
        shap_values = explainer.shap_values(instance, nsamples=self.samples, l1_reg=0.0, silent=True)
        per_coord   = self._per_coordinate(shap_values)

        node_values = per_coord[:, : self.node_features]
        edge_values = per_coord[:, self.node_features:]

        self.node_sum     += np.abs(node_values)
        self.edge_sum     += np.abs(edge_values)
        self.node_overall += np.abs(node_values).sum(axis=0)
        self.edge_overall += np.abs(edge_values).sum(axis=0)

        instance_output  = self._masked_outputs(graph, instance)[0]
        reference        = instance_output - np.asarray(explainer.expected_value, dtype=np.float64)
        attributed       = per_coord.sum(axis=1)
        for index, name in enumerate(self.COORDINATES):
            self.completeness[name] += float(attributed[index] / reference[index]) / len(self.subset_graphs) if reference[index] != 0.0 else 0.0

    def _means(self):
        event_count = len(self.subset_graphs)
        return {
            "coordinates"  : self.COORDINATES,
            "node_overall" : self.node_overall / event_count,
            "edge_overall" : self.edge_overall / event_count,
            "node_per_axis": self.node_sum / event_count,
            "edge_per_axis": self.edge_sum / event_count,
            "completeness" : self.completeness,
            "events"       : event_count,
            "samples"      : self.samples,
        }

    def run(self):
        import shap

        self.logger.section("[Kernel SHAP]")
        self._select_subset()

        background = np.zeros((1, self.total_features), dtype=np.float64)
        instance   = np.ones((1, self.total_features), dtype=np.float64)

        self.model.eval()
        np.random.seed(self.seed)

        with self.logger.track() as progress:
            task_id = progress.add_task("Kernel SHAP", total=len(self.subset_graphs))
            for graph in self.subset_graphs:
                explainer = shap.KernelExplainer(lambda gates, current=graph: self._masked_outputs(current, gates), background, silent=True)
                self._accumulate_event(explainer, graph, instance)
                progress.advance(task_id)

        return self._means()
