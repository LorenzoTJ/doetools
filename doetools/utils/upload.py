from pathlib import Path
import pandas as pd


class FileUploaderMixin:
    
    """
    Mixin providing lightweight CSV/XLSX loading utilities.

    This helper abstracts file loading for design and response data. It supports
    CSV and Excel (``.xlsx``) formats and raises informative errors for missing
    files or unsupported extensions.
    """

    def upload_file(self, path: str | Path):
        
        """
        Load a CSV or XLSX file into a pandas DataFrame.

        Parameters
        ----------
        path : str or Path
            File path to load. Must exist and have extension ``.csv`` or ``.xlsx``.

        Returns
        -------
        pd.DataFrame
            Loaded data.

        Raises
        ------
        FileNotFoundError
            If the provided path does not exist.
        ValueError
            If the file extension is not supported.
            
        """
        
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        
        if path.suffix.lower() == ".xlsx":
            return pd.read_excel(path, engine="openpyxl")
        
        raise ValueError(f"Unsupported file type: {path.suffix}")