"""Public-kit format check, with optional scoring on released validation answers."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from slot_forecasting.io import read_queries, read_submission
from slot_forecasting.rounds import round_metadata, score_round
from slot_forecasting.schema import ValidationError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path, help="Forecast CSV to check; never uploaded")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "public",
                        help="This round's public data folder (default: public/ beside this script)")
    parser.add_argument("--split", choices=("test", "validation"), default="test",
                        help="Test checks format only; validation also reports public round points")
    args = parser.parse_args(argv)
    try:
        metadata = round_metadata(args.data)
        queries = read_queries(args.data / args.split / "queries.csv")
        content = args.submission.read_bytes()
        predictions = read_submission(args.submission, {q.query_id for q in queries}, content=content)
        result = score_round(args.submission, args.data) if args.split == "validation" else None
        if result is not None and args.submission.read_bytes() != content:
            raise ValidationError("Submission changed during checking; retry with the saved file.")
        print(f"VALID FORMAT: {len(predictions)} queries | {metadata['round']} | {args.split}")
        print(f"Season: {metadata['season_id']}")
        print(f"SHA-256: {hashlib.sha256(content).hexdigest()}")
        if result is not None:
            print(f"Validation points: {result['technical_points']:.6f} / {metadata['max_points']}")
            for name, row in result["by_condition"].items():
                print(f"  {name}: loss ratio {row['loss_ratio']:.6f} (lower is better)")
        else:
            print("Test answers are private; this check does not calculate your event score.")
        print("Local check only. Nothing was uploaded. Acceptance and code reproduction are separate.")
        return 0
    except (ValidationError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"CHECK FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
