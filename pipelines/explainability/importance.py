from __future__ import annotations

import numpy as np
import torch


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
