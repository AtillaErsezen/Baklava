import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

from storage3.exceptions import StorageApiError

from scripts.register_dataset import CheckFailed, prepare_record, register


class FakeStorage:
    def __init__(self, files, public_client=False):
        self.files = files
        self.public_client = public_client
        self.bucket = SimpleNamespace(public=False, file_size_limit=1024, allowed_mime_types=None)
        self.denial_code = "not_found"
        self.allow_download = False
        self.upload_calls = 0

    def get_bucket(self, name):
        assert name == "datasets"
        return self.bucket

    def from_(self, name):
        assert name == "datasets"
        return self

    def upload(self, path, raw, file_options):
        assert file_options["upsert"] == "false"
        self.upload_calls += 1
        if path in self.files:
            raise StorageApiError("Exists", "ResourceAlreadyExists", 409)
        self.files[path] = raw

    def download(self, path):
        if self.public_client and not self.allow_download:
            raise StorageApiError("Denied", self.denial_code, 404)
        return self.files[path]


class FakeQuery:
    def __init__(self, rows):
        self.rows = rows
        self.dataset_id = None

    def select(self, columns):
        return self

    def eq(self, field, value):
        assert field == "id"
        self.dataset_id = value
        return self

    def upsert(self, record, on_conflict, ignore_duplicates):
        assert on_conflict == "id" and ignore_duplicates is True
        self.rows.setdefault(record["id"], copy.deepcopy(record))
        return self

    def execute(self):
        return SimpleNamespace(data=[copy.deepcopy(self.rows[self.dataset_id])] if self.dataset_id in self.rows else [])


class FakeClient:
    def __init__(self, rows, files, public_client=False):
        self.rows = rows
        self.storage = FakeStorage(files, public_client)

    def table(self, name):
        assert name == "datasets"
        return FakeQuery(self.rows)


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(__file__).resolve().parents[1] / "supabase/fixtures/setup_demo.csv"
        self.raw = self.path.read_bytes()
        self.url = "https://bsqmjxaivnuapxzrrcen.supabase.co"
        self.record, self.content_type = prepare_record(self.path, self.raw, self.url, "Synthetic churn", "v1", 3, True)
        self.rows, self.files = {}, {}
        self.backend = FakeClient(self.rows, self.files)
        self.browser = FakeClient(self.rows, self.files, public_client=True)

    def perform(self, record=None):
        return register(self.backend, self.browser, record or self.record, self.raw, self.content_type)

    def test_profile_tracks_exact_file_and_changed_bytes_get_new_identity(self):
        self.assertEqual(self.record["source_sha256"], "77c298d794de2393c194530e166c369a5e3346dbdd273f95a5d3bd593060a569")
        self.assertEqual((self.record["file_size_bytes"], self.record["row_count"], self.record["column_count"]), (126, 12, 3))
        self.assertEqual(len(self.record["preview_rows"]), 3)
        default, _ = prepare_record(self.path, self.raw, self.url, "Synthetic churn", "v1")
        changed, _ = prepare_record(self.path, self.raw.replace(b"2,80,1", b"2,81,1"), self.url, "Synthetic churn", "v2")
        self.assertEqual(default["preview_rows"], [])
        self.assertNotEqual(changed["id"], self.record["id"])
        self.assertNotEqual(changed["storage_path"], self.record["storage_path"])

    def test_retry_preserves_one_file_and_record(self):
        self.assertEqual(self.perform(), self.perform())
        self.assertEqual(len(self.rows), 1)
        self.assertEqual(len(self.files), 1)
        self.assertEqual(self.files[self.record["storage_path"]], self.raw)

    def test_interrupted_upload_can_be_registered_on_retry(self):
        self.files[self.record["storage_path"]] = self.raw
        self.perform()
        self.assertEqual(len(self.rows), 1)

    def test_existing_metadata_is_never_rewritten(self):
        self.perform()
        changed = {**self.record, "version_label": "different"}
        with self.assertRaises(CheckFailed):
            self.perform(changed)
        self.assertEqual(self.rows[self.record["id"]]["version_label"], "v1")
        self.assertEqual(self.backend.storage.upload_calls, 1)

    def test_corrupt_existing_file_is_not_registered_or_replaced(self):
        self.files[self.record["storage_path"]] = b"different bytes"
        with self.assertRaises(CheckFailed):
            self.perform()
        self.assertEqual(self.rows, {})
        self.assertEqual(self.files[self.record["storage_path"]], b"different bytes")

    def test_public_bucket_or_oversized_file_is_rejected_before_upload(self):
        self.backend.storage.bucket.public = True
        with self.assertRaises(CheckFailed):
            self.perform()
        self.backend.storage.bucket.public = False
        self.backend.storage.bucket.file_size_limit = 1
        with self.assertRaises(CheckFailed):
            self.perform()
        self.assertEqual(self.backend.storage.upload_calls, 0)

    def test_read_leak_or_service_error_does_not_pass_privacy_check(self):
        self.browser.storage.allow_download = True
        with self.assertRaisesRegex(CheckFailed, "could download"):
            self.perform()
        self.browser.storage.allow_download = False
        self.browser.storage.denial_code = "InternalError"
        with self.assertRaisesRegex(CheckFailed, "inconclusive"):
            self.perform()


if __name__ == "__main__":
    unittest.main()
