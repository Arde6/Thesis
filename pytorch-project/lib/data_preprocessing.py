import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import os
import h5py
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, data_dir: str):
        """
        Initialize the data preprocessor.
        
        Args:
            data_dir (str): Directory containing the raw data files
        """
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.target_name = None
        
    def read_bmespecimen_file(self, file_path: str) -> Tuple[np.array, int]:
        """
        Extracts structured data from a .bmespecimen JSON file.
        
        Args:
            file_path (str): Path to the .bmespecimen file
        
        Returns:
            dict: Processed JSON data containing selected fields
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            raise

        try:
            root = data.get("data", {})
            
            # --- specimenData ---
            specimen_data = root.get("specimenData", {})
            specimen_info = {
                "id": specimen_data.get("id"),
                "label": specimen_data.get("label")
            }

            # --- heaterProfiles ---
            heater_profiles = []
            for hp in root.get("heaterProfiles", []):
                steps = hp.get("steps", [])
                temperatures = [s.get("temperature") for s in steps if "temperature" in s]
                heater_profiles.append({
                    "id": hp.get("id"),
                    "uid": hp.get("uid"),
                    "temperatures": temperatures
                })

            # --- sensorConfigs ---
            sensor_configs = []
            for sc in root.get("sensorConfigs", []):
                sensor_configs.append({
                    "id": sc.get("id"),
                    "sensorId": sc.get("sensorId"),
                    "heaterProfileId": sc.get("heaterProfileId")
                })

            # --- cycles ---
            cycles = []
            for cy in root.get("cycles", []):
                cycles.append({
                    "id": cy.get("id"),
                    "sensorId": cy.get("sensorId")
                })

            # --- dataColumns ---
            data_columns = []
            for dc in root.get("dataColumns", []):
                data_columns.append({
                    "name": dc.get("name"),
                    "key": dc.get("key")
                })

            # Extract only the list of keys for mapping
            column_keys = [dc.get("key") for dc in data_columns if "key" in dc]

            # --- specimenDataPoints ---
            specimen_data_points = root.get("specimenDataPoints", [])

            # Map data points to their corresponding column keys
            mapped_data_points = []

            # Handle case: multiple data points (list of lists)
            if specimen_data_points and isinstance(specimen_data_points[0], list):
                for point in specimen_data_points:
                    point_map = {}
                    for idx, key in enumerate(column_keys):
                        if idx < len(point):
                            point_map[key] = point[idx]
                    mapped_data_points.append(point_map)
            else:
                # Single list of values
                point_map = {}
                for idx, key in enumerate(column_keys):
                    if idx < len(specimen_data_points):
                        point_map[key] = specimen_data_points[idx]
                mapped_data_points = [point_map]
            
            for i in mapped_data_points:
                i["specimen_id"] = specimen_data['id']

            # --- Link specimenDataPoints to cycles by cycle_id ---
            if cycles and mapped_data_points:
                # Create a lookup dictionary for fast access
                cycle_lookup = {c["id"]: c["sensorId"] for c in cycles if "id" in c and "sensorId" in c}

                for dp in mapped_data_points:
                    cycle_id = dp.get("cycle_id")  # match key
                    if cycle_id in cycle_lookup:
                        dp["sensor_id"] = cycle_lookup[cycle_id]
            
            # --- Link specimenDataPoints to sensor_configs by sensor_id ---
            if sensor_configs and mapped_data_points:
                # Create a lookup dictionary for fast access
                sensor_configs_lookup = {c["sensorId"]: c["heaterProfileId"] for c in sensor_configs if "sensorId" in c and "heaterProfileId" in c}

                for dp in mapped_data_points:
                    sensor_id = dp.get("sensor_id")  # match key
                    if sensor_id in sensor_configs_lookup:
                        dp["heater_profile_id"] = sensor_configs_lookup[sensor_id]
            
            # --- Link specimenDataPoints to sensor_configs by cycle_step_index ---
            if heater_profiles and mapped_data_points:
                # Create a lookup dictionary for fast access
                heater_profiles_lookup = {c["id"]: c["temperatures"] for c in heater_profiles if "id" in c and "temperatures" in c}

                for dp in mapped_data_points:
                    index = dp.get("cycle_step_index")
                    heater_profile_id = dp.get("heater_profile_id")  # match key
                    if heater_profile_id in heater_profiles_lookup:
                        dp["heater_temperature"] = heater_profiles_lookup[heater_profile_id][index]
            
            # --- Link specimenDataPoints to specimen_info by sensor_id ---
            for i in mapped_data_points:
                    dp["specimen_id"] = specimen_info['id']

            # --- build final structured output ---
            keys_to_keep = ["resistance_gassensor","temperature","pressure","relative_humidity"]
            extracted_data = [
                #"specimenData": specimen_info,
                #"heaterProfiles": heater_profiles,
                #"sensorConfigs": sensor_configs,
                #"cycles": cycles,
                #"dataColumns": data_columns,
                #"specimenDataPoints": mapped_data_points
                [d[k] for k in keys_to_keep if k in d] for d in mapped_data_points
            ]

            group_size = 10
            result = []
            target= []

            # --- NEW LOGIC: Extract the string label ---
            classification_label_string = specimen_info.get("label", "unknown_class")

            for i in range(0, len(extracted_data), group_size):
                group = extracted_data[i:i+group_size]
                if len(group) < group_size:
                    continue
                
                tenth = group[-1].copy()
                first_values = [lst[0] for lst in group[:-1] if lst]
                features_sample = first_values + tenth
                result.append(features_sample)
                
                # Assign the meaningful string label to every sample
                target.append(classification_label_string) 
            
            features = np.array(result)
            target = np.array(target) # Target is now an array of strings

            return features, target
        
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise

    def process_all_files(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process all .bmespecimen files in the directory.
        
        Returns:
            tuple: (features_array, targets_array) (targets are now encoded integers)
        """
        import glob

        features_all = []
        targets_all = []
        specimen = 0

        file_paths = glob.glob(os.path.join(self.data_dir, "*.bmespecimen"))
        logger.info(f"Found {len(file_paths)} files to process")


        for file_path in file_paths:
                try:
                    features, target = self.read_bmespecimen_file(file_path)
                    
                    # --- REMOVED faulty target indexing (the 'specimen' loop) ---
                    
                    features_all.extend(features)
                    targets_all.extend(target)
                    specimen = specimen+1
                except Exception as e:
                    logger.warning(f"Skipping file {file_path} due to error: {str(e)}")
                    continue
        
        if not features_all:
            raise ValueError("No valid data files were processed")

        features_final = np.array(features_all)
        targets_string = np.array(targets_all)

        # --- FIX HDF5 TypeError: Apply Label Encoding to convert strings to integers ---
        if targets_string.dtype.kind in ('U', 'S', 'O'): 
            logger.info(f"Applying Label Encoding to string targets of dtype: {targets_string.dtype}")
            # The LabelEncoder will convert the string labels (e.g., 'ClassA', 'ClassB') to integers (0, 1)
            targets_encoded = self.label_encoder.fit_transform(targets_string.ravel()).reshape(-1, 1) # Reshape back to 2D
        else:
            targets_encoded = targets_string
        
        return features_final, targets_encoded # targets are now integers


    def save_processed_data(self, 
                          features: np.ndarray, 
                          targets: np.ndarray, 
                          output_path: str):
        """
        Save processed data to an HDF5 file.
        
        Args:
            features (np.ndarray): Processed features
            targets (np.ndarray): Processed targets
            output_path (str): Path to save the HDF5 file
        """
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('features', data=features)
            f.create_dataset('targets', data=targets)
            
            # Save class names if LabelEncoder was used
            if hasattr(self, 'label_encoder') and hasattr(self.label_encoder, 'classes_'):
                class_names = [name.encode('utf-8') for name in self.label_encoder.classes_]
                f.create_dataset('class_names', data=class_names)

            # Save metadata
            metadata = {
                'num_samples': len(features),
                'feature_dim': features.shape[1],
                'target_dim': targets.shape[1] if len(targets.shape) > 1 else 1
            }
            for key, value in metadata.items():
                f.attrs[key] = value
            # If scaler has been fitted, save scaler arrays too
            if hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_'):
                try:
                    f.create_dataset('scaler_mean', data=self.scaler.mean_)
                    f.create_dataset('scaler_scale', data=self.scaler.scale_)
                except Exception:
                    # best-effort: ignore scaler saving failures
                    logger.warning('Failed to write scaler arrays to HDF5')

    def save_scaler(self, path: str):
        """
        Persist the fitted scaler mean and scale to a .npz file.
        """
        if not hasattr(self.scaler, 'mean_') or not hasattr(self.scaler, 'scale_'):
            raise RuntimeError('Scaler has not been fitted yet; cannot save scaler.')
        np.savez(path, mean=self.scaler.mean_, scale=self.scaler.scale_)
    
    def load_processed_data(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load processed data from an HDF5 file.
        
        Args:
            file_path (str): Path to the HDF5 file
            
        Returns:
            tuple: (features, targets, class_names)
        """
        class_names = []
        with h5py.File(file_path, 'r') as f:
            features = f['features'][:]
            targets = f['targets'][:]
            if 'class_names' in f:
                class_names = [name.decode('utf-8') for name in f['class_names'][:]]

        return features, targets, class_names
    
    def prepare_training_data(self, 
                            features: np.ndarray, 
                            targets: np.ndarray,
                            train_split: float = 0.8,
                            random_seed: int = 42) -> Dict[str, np.ndarray]:
        # Set random seed
        np.random.seed(random_seed)

        # Split data
        indices = np.random.permutation(len(features))
        split_idx = int(len(features) * train_split)
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        # --- FIX: Scale features ONLY using training set statistics ---
        
        # 1. Fit scaler on TRAINING features
        train_features_scaled = self.scaler.fit_transform(features[train_indices])
        
        # 2. Transform VALIDATION features using the *fitted* scaler
        val_features_scaled = self.scaler.transform(features[val_indices])
        
        # Create train and validation sets
        train_data = {
            'features': train_features_scaled, # Use already scaled training features
            'targets': targets[train_indices] 
        }
        
        val_data = {
            'features': val_features_scaled, # Use already scaled validation features
            'targets': targets[val_indices] 
        }
        
        return train_data, val_data
    
    #def save_train_val_to_text(self, train_data: dict, val_data: dict,
    #                           train_path: str = "train_data.txt",
    #                           val_path: str = "val_data.txt"):
    #    """
    #    Save training and validation data (features + targets) to text files.
    #
    #    Args:
    #        train_data (dict): Dictionary containing 'features' and 'targets' for training.
    #        val_data (dict): Dictionary containing 'features' and 'targets' for validation.
    #        train_path (str): File path for training data output.
    #        val_path (str): File path for validation data output.
    #    """
    #    # Combine features and targets for saving
    #    train_combined = np.hstack((train_data["features"], train_data["targets"]))
    #    val_combined = np.hstack((val_data["features"], val_data["targets"]))
    #
    #    np.savetxt(train_path, train_combined, fmt="%.6f")
    #    np.savetxt(val_path, val_combined, fmt="%.6f")
    #
    #    logger.info(f"Training data saved to: {train_path}")
    #    logger.info(f"Validation data saved to: {val_path}")