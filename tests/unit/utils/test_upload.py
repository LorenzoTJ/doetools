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


@pytest.mark.parametrize("extension", ["csv", "xlsx", "CSV", "XLSX"])
@pytest.mark.parametrize("string_path", [False, True])
def test_export_file_round_trip(tmp_path, extension, string_path):
    uploader = TestUploader()
    frame = pd.DataFrame(
        {"Temperatura": [20.5, 30.25], "Qualit\u00e0": ["caff\u00e8, t\u00e8", '"ottimo"\nseconda riga']},
        index=[10, 20],
    )
    path = tmp_path / f"table.{extension}"
    destination = str(path) if string_path else path

    uploader._export_file(frame, destination)

    loaded = uploader.upload_file(destination)
    pd.testing.assert_frame_equal(loaded, frame.reset_index(drop=True))
    if extension.lower() == "csv":
        assert "Qualit\u00e0" in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("filename", ["table", "table.txt", "table.xls"])
def test_export_file_rejects_unsupported_extension(tmp_path, filename):
    path = tmp_path / filename
    with pytest.raises(ValueError, match=r"Supported formats: \.csv, \.xlsx"):
        TestUploader()._export_file(pd.DataFrame({"A": [1]}), path)
    assert not path.exists()


def test_export_file_rejects_unknown_destination_type():
    with pytest.raises(TypeError, match="destination must be a path"):
        TestUploader()._export_file(pd.DataFrame({"A": [1]}), 123)
