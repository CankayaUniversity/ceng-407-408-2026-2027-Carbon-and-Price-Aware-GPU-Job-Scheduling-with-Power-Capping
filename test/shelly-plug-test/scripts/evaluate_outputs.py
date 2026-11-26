"""Evaluate saved task outputs outside the measured job."""

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import requests

from run_tasks import (MAX_TOKENS, REQUEST_DEFAULTS, load_labels, messages_for,
                       positive_float, read_jsonl, request_completion)


def parse_label(text, labels):
    cleaned = text.strip().strip('"\'“”‘’. \t\r\n').casefold()
    return next((label for label in labels if label.casefold() == cleaned), None)


def classification_metrics(truth, predictions, labels):
    matrix = {label: {column: 0 for column in [*labels, "<unparseable>"]} for label in labels}
    for expected, predicted in zip(truth, predictions):
        if expected not in labels:
            raise ValueError(f"Unknown ground-truth product: {expected}")
        matrix[expected][predicted or "<unparseable>"] += 1
    scores = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in labels if other != label)
        fn = sum(matrix[label].values()) - tp
        scores[label] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "accuracy": sum(a == b for a, b in zip(truth, predictions)) / len(truth) if truth else None,
        "macro_f1": statistics.mean(scores.values()), "f1_by_label": scores,
        "confusion_matrix": matrix, "n_unparseable": predictions.count(None),
        "macro_f1_definition": "Mean over all eight labels; zero when a label has no support or predictions",
    }


def distribution(values):
    return {"n": len(values), "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "counts": dict(sorted(Counter(values).items()))}


def sentence_count(text):
    # A transparent heuristic, not a linguistic quality assessment. Keep decimals intact.
    text = re.sub(r"(?<=\d)\.(?=\d)", "<decimal>", text.strip())
    return len([part for part in re.split(r"[.!?]+(?:[\"'”’]*\s+|[\"'”’]*$)", text) if part.strip()])


def duplicate_groups(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row["output"])].append(row["id"])
    return [ids for ids in groups.values() if len(ids) > 1]


def evaluate(args):
    inputs = {row["id"]: row for row in read_jsonl(args.input)}
    rows = read_jsonl(args.outputs)
    tasks = {row.get("task") for row in rows}
    if len(tasks) != 1 or next(iter(tasks)) not in MAX_TOKENS:
        raise ValueError("Output rows must all contain the same valid task")
    task = next(iter(tasks))
    if args.reclassify and task == "classify":
        raise ValueError("--reclassify is for summarize or synth")
    if args.task and args.task != task:
        raise ValueError("--task differs from output rows")
    for row in rows:
        if row["id"] not in inputs:
            raise ValueError(f"Output id missing from input: {row['id']}")
        if not isinstance(row.get("output"), str):
            raise ValueError(f"{row['id']}: output must be a string")
    labels_path = Path(args.labels) if args.labels else Path(args.input).with_name("labels.json")
    labels = load_labels(labels_path) if task == "classify" or args.reclassify else None
    valid = [row for row in rows if not row.get("error") and row["output"].strip()]
    valid_ids = {row["id"] for row in valid}
    report = {"task": task, "input_file": args.input, "outputs_file": args.outputs,
              "n_input": len(inputs), "n_outputs": len(rows), "n_evaluated": len(valid),
              "n_failed": len(rows) - len(valid), "n_missing": len(inputs) - len(rows)}
    if task == "classify":
        report.update(classification_metrics(
            [inputs[row["id"]]["product"] for row in rows],
            [parse_label(row["output"], labels) if row["id"] in valid_ids else None for row in rows], labels))
    else:
        report["word_count"] = distribution([len(row["output"].split()) for row in valid])
        report["character_count"] = distribution([len(row["output"]) for row in valid])
        if task == "summarize":
            report["sentence_count"] = distribution([sentence_count(row["output"]) for row in valid])
            report["sentence_count_method"] = "Punctuation boundary heuristic; decimals preserved; abbreviations may overcount"
        else:
            report["n_within_120_200_words"] = sum(120 <= len(row["output"].split()) <= 200 for row in valid)
            report["exact_duplicate_groups"] = duplicate_groups(valid, lambda text: text)
            report["near_duplicate_groups"] = duplicate_groups(
                valid, lambda text: re.sub(r"\s+", " ", text.casefold()).strip()[:200])
            report["near_duplicate_method"] = "First 200 characters after casefolding and whitespace normalization"
        if args.reclassify:
            truth, predictions, failures = [], [], []
            with requests.Session() as session:
                for row in valid:
                    truth.append(inputs[row["id"]]["product"])
                    try:
                        payload = {**REQUEST_DEFAULTS, "max_tokens": MAX_TOKENS["classify"],
                                   "messages": messages_for("classify", {"id": row["id"], "text": row["output"]}, labels)}
                        if args.model:
                            payload["model"] = args.model
                        result = request_completion(session, args.server_url, payload, args.timeout)
                        if "<think>" in result["output"].casefold() or result["reasoning_chars"]:
                            raise RuntimeError("Reclassification contains <think>")
                        predictions.append(parse_label(result["output"], labels))
                    except RuntimeError as exc:
                        predictions.append(None)
                        failures.append({"id": row["id"], "error": str(exc)})
            report["reclassification"] = classification_metrics(truth, predictions, labels)
            report["reclassification"]["failures"] = failures
    destination = Path(str(args.outputs) + "_eval.json")
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Metric                      Value")
    for key in ("task", "n_outputs", "n_evaluated", "n_failed", "n_missing", "accuracy", "macro_f1", "n_unparseable"):
        if key in report:
            print(f"{key:28}{report[key]}")
    for key in ("sentence_count", "word_count"):
        if key in report:
            print(f"{key + ' mean':28}{report[key]['mean']}")
    if "exact_duplicate_groups" in report:
        print(f"{'exact duplicate groups':28}{len(report['exact_duplicate_groups'])}")
        print(f"{'near duplicate groups':28}{len(report['near_duplicate_groups'])}")
    if "reclassification" in report:
        print(f"{'reclassify accuracy':28}{report['reclassification']['accuracy']}")
    print(destination)
    return 1 if report.get("reclassification", {}).get("failures") else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs")
    parser.add_argument("--input", required=True)
    parser.add_argument("--task", choices=MAX_TOKENS, help="Optional check; otherwise inferred from outputs")
    parser.add_argument("--labels")
    parser.add_argument("--reclassify", action="store_true")
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=positive_float, default=300)
    args = parser.parse_args(argv)
    if args.reclassify and args.task == "classify":
        parser.error("--reclassify is for summarize or synth")
    try:
        return evaluate(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
