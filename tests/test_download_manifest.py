from cho2017_benchmark.data import download as dl


def test_file_manifest_columns():
    assert dl.FILE_MANIFEST_COLUMNS[0] == "subject_id"
    for col in ("download_status", "validation_status", "downloaded_at", "error_message"):
        assert col in dl.FILE_MANIFEST_COLUMNS


def test_subject_url_format():
    assert dl.subject_url(1).endswith("/s01.mat")
    assert dl.subject_url(52).endswith("/s52.mat")
    assert dl.subject_id_str(5) == "s05"


def test_validate_mat_header(tiny_mat):
    ok, msg = dl.validate_mat_header(tiny_mat)
    assert ok, msg


def test_validate_mat_header_missing(tmp_path):
    ok, msg = dl.validate_mat_header(tmp_path / "nope.mat")
    assert not ok and msg


def test_disk_check(tmp_path):
    chk = dl.check_disk_space(tmp_path, 1)
    assert chk.ok and chk.free_bytes > 0 and chk.needed_bytes >= 1
