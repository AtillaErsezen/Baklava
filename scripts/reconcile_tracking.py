"""Reapply a durable tracking log in order, using stable event IDs for safe retries."""
import argparse
from tracking import reconcile_log, supabase_client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    args = parser.parse_args()
    client = supabase_client()
    if client is None:
        parser.error("Load SUPABASE_URL and SUPABASE_KEY using --env-file .env.")
    try:
        count = reconcile_log(args.log, client)
    except Exception:
        raise SystemExit("Reconciliation stopped. Earlier events may be applied; the log is unchanged and safe to retry. Check the migration, credentials, and event format.") from None
    print(f"Reconciled {count} events; repeated event IDs create no duplicates.")


if __name__ == "__main__":
    main()
