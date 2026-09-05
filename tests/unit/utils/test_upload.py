import pandas as pd
import pytest

from doetools.utils.upload import FileUploaderMixin


class TestUploader(FileUploaderMixin):
    """Concrete class to expose the mixin method for testing."""
    pass

def test_upload_file_csv(tmp_path):
    uploader = TestUploader()
    data = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    csv_path = tmp_path / "sample.csv"
    data.to_csv(csv_path, index=False)

    loaded = uploader.upload_file(csv_path)

    pd.testing.assert_frame_equal(loaded, data)

def test_upload_file_xlsx(tmp_path):
    uploader = TestUploader()
    data = pd.DataFrame({"A": [3, 4], "B": ["u", "v"]})
    xlsx_path = tmp_path / "sample.xlsx"
    data.to_excel(xlsx_path, index=False)

    loaded = uploader.upload_file(xlsx_path)

    pd.testing.assert_frame_equal(loaded, data)

def test_upload_file_dataframe_returns_defensive_copy():
    uploader = TestUploader()
    data = pd.DataFrame({"A": [1, 2]}, index=[10, 20])

    loaded = uploader.upload_file(data)
    loaded.loc[10, "A"] = 99

    assert data.loc[10, "A"] == 1
    assert list(loaded.index) == [10, 20]

def test_upload_file_not_found(tmp_path):
    uploader = TestUploader()
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        uploader.upload_file(missing_path)

def test_upload_file_unsupported_extension(tmp_path):
    uploader = TestUploader()
    bad_path = tmp_path / "data.txt"
    bad_path.write_text("hello")

    with pytest.raises(ValueError, match="Unsupported file type"):
        uploader.upload_file(bad_path)

def test_upload_file_rejects_unknown_source_type():
    uploader = TestUploader()

    with pytest.raises(TypeError, match="source must be"):
        uploader.upload_file(123)
