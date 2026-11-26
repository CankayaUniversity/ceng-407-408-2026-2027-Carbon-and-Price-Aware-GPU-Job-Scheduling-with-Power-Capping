"""Compare generated text by id, independently of file order and timings."""

import argparse
import sys

from run_tasks import read_jsonl


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a")
    parser.add_argument("b")
    args = parser.parse_args(argv)
    try:
        left = {row["id"]: row for row in read_jsonl(args.a)}
        right = {row["id"]: row for row in read_jsonl(args.b)}
        same = 0
        valid = True
        for item_id in sorted(left.keys() | right.keys()):
            a, b = left.get(item_id), right.get(item_id)
            if a is None or b is None:
                print(f"{item_id}: missing in {'A' if a is None else 'B'}")
                valid = False
                continue
            if any(not isinstance(row.get("output"), str) or not row["output"].strip()
                   or row.get("error") for row in (a, b)):
                print(f"{item_id}: invalid or failed output")
                valid = False
                continue
            if a["output"] == b["output"]:
                same += 1
            else:
                print(f"{item_id}\n  A: {a['output'][:120]!r}\n  B: {b['output'][:120]!r}")
        print(f"Identical outputs: {same}/{len(left.keys() | right.keys())}")
        return 0 if valid and same == len(left) == len(right) else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
