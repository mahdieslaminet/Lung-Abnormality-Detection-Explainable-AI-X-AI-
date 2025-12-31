import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import os

class DatasetLoader:
    """
    A class to handle dynamic loading of datasets using KaggleHub.
    Includes smart fallback for non-tabular datasets (like image directories).
    """
    
    def load_dataset(self, dataset_handle: str, file_path: str = "", **kwargs):
        """
        Loads a dataset. If it's a CSV/Table, returns the content.
        If it's a collection of files (images), returns a DataFrame listing the files.
        """
        print(f"Processing dataset: {dataset_handle}, Requested File: {file_path}")
        
        # 1. Always ensure the dataset is downloaded first to inspect it
        try:
            # dataset_download returns the local path to the dataset folders
            local_path = kagglehub.dataset_download(dataset_handle)
            print(f"Dataset downloaded to: {local_path}")
        except Exception as e:
            print(f"Download failed: {e}")
            raise e

        # 2. If a specific valid tabular file is requested, try to load it using the Adapter
        if file_path and (file_path.endswith('.csv') or file_path.endswith('.json')):
             try:
                return kagglehub.load_dataset(
                    KaggleDatasetAdapter.PANDAS,
                    dataset_handle,
                    file_path,
                    **kwargs
                )
             except Exception as e:
                 print(f"Adapter load failed: {e}")
                 # Fall through to listing

        # 3. If no file specified, check if there's exactly one obvious CSV to load
        if not file_path:
             csv_files = []
             for root, dirs, files in os.walk(local_path):
                 for f in files:
                     if f.endswith('.csv'):
                         csv_files.append(os.path.join(root, f))
             
             if len(csv_files) == 1:
                 # Auto-select the single CSV
                 rel_path = os.path.relpath(csv_files[0], local_path)
                 print(f"Auto-detected single CSV: {rel_path}")
                 return kagglehub.load_dataset(
                    KaggleDatasetAdapter.PANDAS,
                    dataset_handle,
                    rel_path,
                    **kwargs
                )

        # 4. Fallback: If it's an image dataset or ambiguous, return a listing of files
        # This allows the frontend to at least show "Contains 500 images" etc.
        print("Creating file listing for non-tabular/ambiguous dataset...")
        return self._create_file_listing_dataframe(local_path)

    def _create_file_listing_dataframe(self, local_path):
        """
        Scans the directory and returns a DataFrame of files.
        """
        files_data = []
        MAX_FILES = 50 # Limit for preview
        
        for root, dirs, files in os.walk(local_path):
            for name in files:
                if len(files_data) >= MAX_FILES:
                    break
                    
                full_path = os.path.join(root, name)
                rel_path = os.path.relpath(full_path, local_path)
                size = os.path.getsize(full_path)
                
                files_data.append({
                    "Filename": name,
                    "Type": os.path.splitext(name)[1],
                    "Size (KB)": round(size / 1024, 2),
                    "Path": rel_path
                })
            if len(files_data) >= MAX_FILES:
                break
                
        if not files_data:
             return pd.DataFrame([{"Info": "Empty Dataset or Directory"}])
             
        return pd.DataFrame(files_data)

    def get_dataset_file_path(self, dataset_handle: str, relative_path: str):
        """
        Returns the absolute path of a file within a dataset.
        """
        try:
            # This returns the cached path instantly if already downloaded
            local_path = kagglehub.dataset_download(dataset_handle)
            full_path = os.path.join(local_path, relative_path)
            if os.path.exists(full_path):
                return full_path
            else:
                raise FileNotFoundError(f"File not found: {full_path}")
        except Exception as e:
            raise e

    def get_preview(self, df: pd.DataFrame, rows: int = 5):
        """
        Returns a dictionary preview of the dataframe.
        """
        if df is None:
            return None
        
        # Ensure we don't return too much data
        preview_df = df.head(rows)
        
        # Handle NaN values for JSON serialization
        preview_df = preview_df.fillna("")
        
        return {
            "columns": list(preview_df.columns),
            "shape": df.shape,
            "head": preview_df.to_dict(orient='records')
        }
