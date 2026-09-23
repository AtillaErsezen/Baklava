"""Upload an approved CSV/Parquet and register its exact bytes in the shared demo.

Run as a module from the repository root with uv run --frozen --env-file .env.
Metadata and any opted-in preview rows are publicly readable. File bytes stay private.
"""

import argparse
import fnmatch
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import uuid

import pandas as pd
from storage3.exceptions import StorageApiError
from supabase import ClientOptions, create_client


class CheckFailed(Exception):
    """A safe, actionable message that contains no credentials or raw responses."""


def configuration(environ):
    url = environ.get("SUPABASE_URL", "").strip().rstrip("/")
    if not re.fullmatch(r"https://[a-z0-9]{20}\.supabase\.co", url):
        raise CheckFailed("Set SUPABASE_URL to your hosted project's https://<ref>.supabase.co URL.")
    keys = {}
    for label, variable, prefix in (
        ("Publishable key", "SUPABASE_PUBLISHABLE_KEY", "sb_publishable_"),
        ("Backend key", "SUPABASE_KEY", "sb_secret_"),
    ):
        key = environ.get(variable, "").strip()
        if not key.startswith(prefix) or "REPLACE_ME" in key or len(key) <= len(prefix):
            raise CheckFailed(f"Set {variable} using the {prefix} key from Settings > API Keys.")
        keys[label] = key
    return url, keys


def prepare_record(path, raw, url, name, version, preview_rows=0, demo_fixture=False):
    if not name.strip() or not version.strip() or not 0 <= preview_rows <= 10:
        raise CheckFailed("Provide a dataset name, version label, and a preview size from 0 to 10.")
    extension = path.suffix.lower()
    if extension == ".csv":
        frame = pd.read_csv(io.BytesIO(raw))
        content_type = "text/csv"
    elif extension == ".parquet":
        frame = pd.read_parquet(io.BytesIO(raw))
        content_type = "application/vnd.apache.parquet"
    else:
        raise CheckFailed("Choose a CSV or Parquet file.")
    if len(frame.columns) == 0 or not frame.columns.is_unique:
        raise CheckFailed("The dataset must have distinct named columns.")
    digest = hashlib.sha256(raw).hexdigest()
    filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", path.name)
    object_path = f"{digest}/{filename}"
    record = {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{url}/storage/v1/object/datasets/{object_path}")),
        "name": name.strip(),
        "original_filename": path.name,
        "version_label": version.strip(),
        "storage_bucket": "datasets",
        "storage_path": object_path,
        "source_sha256": digest,
        "file_size_bytes": len(raw),
        "row_count": len(frame),
        "column_count": len(frame.columns),
        "column_schema": [{"name": str(column), "dtype": str(dtype)} for column, dtype in frame.dtypes.items()],
        "preview_rows": json.loads(frame.head(preview_rows).to_json(orient="records", date_format="iso")),
        "is_demo_fixture": demo_fixture,
    }
    return record, content_type


def matching_record(rows, expected):
    return len(rows) == 1 and all(rows[0].get(key) == value for key, value in expected.items())


def dataset_rows(client, dataset_id):
    return client.table("datasets").select("*").eq("id", dataset_id).execute().data


def register(backend, browser, record, raw, content_type):
    bucket = backend.storage.get_bucket("datasets")
    if bucket.public:
        raise CheckFailed("The datasets bucket must be private before uploading.")
    if bucket.file_size_limit is not None and len(raw) > bucket.file_size_limit:
        raise CheckFailed("The file exceeds the bucket's configured size limit.")
    if bucket.allowed_mime_types and not any(
        fnmatch.fnmatchcase(content_type, allowed) for allowed in bucket.allowed_mime_types
    ):
        raise CheckFailed("The bucket's allowed content types do not include this file format.")

    # Detect metadata conflicts before any writes. The same bytes/name identify a version.
    existing = dataset_rows(backend, record["id"])
    if existing and not matching_record(existing, record):
        raise CheckFailed("This file is already registered with different metadata; its history was preserved.")
    storage = backend.storage.from_("datasets")
    try:
        storage.upload(record["storage_path"], raw, file_options={
            "content-type": content_type, "upsert": "false", "cache-control": "0",
        })
    except StorageApiError as exc:
        if exc.code not in {"Duplicate", "already_exists", "ResourceAlreadyExists", "KeyAlreadyExists"}:
            raise
        # A retry can find the earlier upload. Never overwrite it; verify its bytes below.
    downloaded = storage.download(record["storage_path"])
    if hashlib.sha256(downloaded).hexdigest() != record["source_sha256"]:
        raise CheckFailed("Stored file hash differs from the local file; nothing was overwritten.")

    # Ignore repeated inserts, never update an immutable dataset record.
    backend.table("datasets").upsert(record, on_conflict="id", ignore_duplicates=True).execute()
    if not matching_record(dataset_rows(backend, record["id"]), record):
        raise CheckFailed("Dataset registration could not be verified. The uploaded file was retained for retry.")
    if not matching_record(dataset_rows(browser, record["id"]), record):
        raise CheckFailed("Public dataset metadata could not be read back correctly.")

    try:
        browser.storage.from_("datasets").download(record["storage_path"])
    except StorageApiError as exc:
        if exc.code not in {"NoSuchKey", "NoSuchBucket", "not_found", "AccessDenied", "unauthorized"}:
            raise CheckFailed("Public download check was inconclusive; inspect Storage access.") from None
    else:
        raise CheckFailed("Public client could download the private file. Review Storage read policies.")
    return record["id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--preview-rows", type=int, default=0, help="0–10 approved rows to expose in public metadata; default 0")
    parser.add_argument("--demo-fixture", action="store_true", help="Label synthetic setup data")
    args = parser.parse_args()
    stage = "local file preparation"
    try:
        url, keys = configuration(os.environ)
        raw = args.file.read_bytes()
        record, content_type = prepare_record(
            args.file, raw, url, args.name, args.version, args.preview_rows, args.demo_fixture,
        )
        options = ClientOptions(postgrest_client_timeout=15, storage_client_timeout=30)
        backend = create_client(url, keys["Backend key"], options=options)
        browser = create_client(url, keys["Publishable key"], options=ClientOptions(postgrest_client_timeout=15, storage_client_timeout=30))
        stage = "Storage upload, registration, and access verification"
        dataset_id = register(backend, browser, record, raw, content_type)
        print(json.dumps({
            "dataset_id": dataset_id,
            "storage_bucket": "datasets",
            "storage_path": record["storage_path"],
            "source_sha256": record["source_sha256"],
            "rows": record["row_count"],
            "columns": record["column_count"],
            "bytes": record["file_size_bytes"],
            "is_demo_fixture": record["is_demo_fixture"],
        }, indent=2))
        print("PASS: private bucket; backend download matches source hash; dataset metadata is readable; public client download denied.")
        print("No training was run. Repeating this command preserves the same dataset ID and file.")
        return 0
    except CheckFailed as exc:
        print(f"REGISTRATION FAILED: {exc}", file=sys.stderr)
    except Exception:
        # Do not leak credentials, signed URLs, or sensitive values from parser/API exceptions.
        print(f"REGISTRATION FAILED during {stage}. Raw error details withheld. Files/rows may exist; rerun the identical command after resolving the issue.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
