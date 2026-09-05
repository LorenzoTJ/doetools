from pathlib import Path
import pandas as pd


class FileUploaderMixin:
    
    """
    Mixin providing lightweight DataFrame/CSV/XLSX loading utilities.

    This helper copies in-memory DataFrames or loads CSV and Excel (``.xlsx``)
    data, raising informative errors for missing files or unsupported extensions.
    """

    def upload_file(self, source: str | Path | pd.DataFrame) -> pd.DataFrame:
        
        """
        Load a CSV or XLSX file into a pandas DataFrame.

        Parameters
        ----------
        source : str, Path, or pandas.DataFrame
            DataFrame to copy, or a file path with extension ``.csv`` or
            ``.xlsx``.

        Returns
        -------
        pd.DataFrame
            Loaded data or a defensive copy of ``source``.

        Raises
        ------
        FileNotFoundError
            If the provided path does not exist.
        TypeError
            If ``source`` is not a path or DataFrame.
        ValueError
            If the file extension is not supported.
            
        """
        
        if isinstance(source, pd.DataFrame):
            return source.copy(deep=True)

        if not isinstance(source, (str, Path)):
            raise TypeError("source must be a path or pandas DataFrame.")

        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(path)

        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        
        if path.suffix.lower() == ".xlsx":
            return pd.read_excel(path, engine="openpyxl")
        
        raise ValueError(f"Unsupported file type: {path.suffix}")
