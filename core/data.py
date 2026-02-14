from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from .config import config
from .logger import Logger
import ray # type: ignore


class DataLoader:
    def __init__(self, logger: Logger = None):
        self.logger = logger

    def parse(self, file_path: Path):
        filename = file_path.stem
        parts = filename.split('_')
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])
        coordinates = (x, y, z)
   
        dataframe = pd.read_csv(
            file_path,
            skiprows=config.data.header_rows_to_skip,
            names=["bin", "x", "y", "z", config.data.light_column],
            comment="b",
            dtype={"x": float, "y": float, "z": float, config.data.light_column: float},
        )
        return dataframe, coordinates

    def load(self, input_directory=None):
        input_dir = Path(input_directory) if input_directory else config.data.input_dir
        file_list = sorted(Path(input_dir).glob("*.csv"))[: config.data.max_files]
        
        @ray.remote
        def parse_file_remote(file_path_str):
            file_path = Path(file_path_str)
            filename = file_path.stem
            parts = filename.split('_')
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            coordinates = (x, y, z)
            
            dataframe = pd.read_csv(
                file_path,
                skiprows=config.data.header_rows_to_skip,
                names=["bin", "x", "y", "z", config.data.light_column],
                comment="b",
                dtype={"x": float, "y": float, "z": float, config.data.light_column: float},
            )
            return dataframe, coordinates
        
        self.logger.section("[Data Loading]")
        self.logger.info(f"Loading {len(file_list)} files in parallel with Ray")
        futures = [parse_file_remote.remote(str(fp)) for fp in file_list]

        results = []
        batch_size = min(50, max(10, len(futures) // 20)) 
        with tqdm(total=len(futures), desc="Loading files") as pbar:
            while futures:
                ready, futures = ray.wait(futures, num_returns=min(batch_size, len(futures)))
                results.extend(ray.get(ready))
                pbar.update(len(ready))
        
        dataframes      = [df for df, _ in results]
        coordinate_rows = [coord for _, coord in results]
            
        coordinate_dataframe = pd.DataFrame(coordinate_rows, columns=list(config.data.coordinate_columns))
        self.logger.info(f"Loaded {len(dataframes)} dataframes with coordinates")
        return dataframes, coordinate_dataframe


class DataProcessor:
    def __init__(self, logger: Logger = None):
        self.logger = logger
    
    def detect_spatial_outliers(self, dataframes, zscore_threshold, radius):
        self.logger.section("[Spatial Outlier Detection]")
        self.logger.info(f"zscore_threshold={zscore_threshold:.2f} | radius={radius:.2f}")
        
        @ray.remote
        def detect_outliers_single(df, zscore_threshold, radius):
            modified = df.copy()
            coordinates = modified[["x", "y", "z"]].values
            tree = cKDTree(coordinates)
            light_values = modified["light_amount"].values.copy()
            outlier_indices = []
            all_neighbors = tree.query_ball_point(coordinates, r=radius, p=2.0)
            
            for sensor_index, neighbor_indices in enumerate(all_neighbors):
                neighbor_indices = [n for n in neighbor_indices if n != sensor_index]
                
                if neighbor_indices:
                    neighbor_values = light_values[neighbor_indices]
                    mean, std = neighbor_values.mean(), neighbor_values.std()
                    
                    if std > 0 and abs(light_values[sensor_index] - mean) / std > zscore_threshold:
                        light_values[sensor_index] = mean
                        outlier_indices.append(sensor_index)
            
            modified["light_amount"] = light_values
            return modified, df.iloc[outlier_indices] if outlier_indices else pd.DataFrame()
        
        chunk_size   = 100  
        self.logger.info(f"Processing {len(dataframes)} dataframes in chunks of {chunk_size}")

        all_cleaned  = []
        all_outliers = []
        
        for i in range(0, len(dataframes), chunk_size):
            chunk = dataframes[i:i + chunk_size]
            futures = [detect_outliers_single.remote(df, zscore_threshold, radius) for df in chunk]
            
            results = []
            batch_size = min(10, max(5, len(futures) // 10))
            with tqdm(total=len(futures), desc=f"Detecting outliers ({i//chunk_size + 1}/{(len(dataframes) + chunk_size - 1)//chunk_size})", leave=False) as pbar:
                while futures:
                    ready, futures = ray.wait(futures, num_returns=min(batch_size, len(futures)), timeout=10.0)
                    results.extend(ray.get(ready))
                    pbar.update(len(ready))
            
            all_cleaned.extend([c for c, _ in results])
            all_outliers.extend([o for _, o in results])

        self.logger.info(f"Completed outlier detection. Cleaned: {len(all_cleaned)} | Outliers: {sum(len(o) for o in all_outliers)}")
        return all_cleaned, all_outliers

    def scale_light_intensity(self, dataframes, scale_factor):
        self.logger.section("[Scaling Light Intensity]")
        self.logger.info(f"scale_factor={scale_factor:.2f}")
        
        @ray.remote
        def scale_single(df, factor):
            modified = df.copy()
            modified["light_amount"] *= factor
            return modified
        
        futures = [scale_single.remote(df, scale_factor) for df in dataframes]
        
        scaled = []
        batch_size = min(50, max(10, len(futures) // 20))
        self.logger.info(f"Scaling {len(dataframes)} dataframes using batch size {batch_size}")

        with tqdm(total=len(futures), desc="Scaling intensity") as pbar:
            while futures:
                ready, futures = ray.wait(futures, num_returns=min(batch_size, len(futures)))
                scaled.extend(ray.get(ready))
                pbar.update(len(ready))
        
        self.logger.info(f"Completed scaling of light intensity for {len(scaled)} dataframes")
        return scaled
    
    def apply_detection_efficiency(self, dataframes, efficiency, seed=42):
        self.logger.section("[Applying Detection Efficiency]")
        self.logger.info(f"efficiency={efficiency:.2f} | seed={seed}")
        
        processed = []
        for idx, df in enumerate(tqdm(dataframes, desc="Applying efficiency")):
            random_generator = np.random.default_rng(seed + idx)
            modified = df.copy()
            trials = modified["light_amount"].round().astype(int).clip(0)
            modified["light_amount"] = random_generator.binomial(trials, efficiency)
            processed.append(modified)
        
        self.logger.info(f"Completed applying detection efficiency for {len(processed)} dataframes")
        return processed

    def apply_log_transform(self, dataframes):
        self.logger.section("[Applying Log Transform]")
        self.logger.info("Transform: log1p to light column")
        
        updated = []
        for index, dataframe in enumerate(tqdm(dataframes, desc="Applying log1p", total=len(dataframes))):
            modified = dataframe.copy()
            modified["light_amount"] = np.log1p(np.maximum(modified["light_amount"].fillna(0), 0))
            updated.append(modified)
        
        self.logger.info(f"Completed log transform for {len(updated)} dataframes")
        return updated


__all__ = [
    "DataLoader",
    "DataProcessor",
]
