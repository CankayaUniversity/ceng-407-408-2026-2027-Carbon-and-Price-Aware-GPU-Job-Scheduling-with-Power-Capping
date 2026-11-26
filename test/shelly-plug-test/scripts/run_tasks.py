"""Sequential, reproducible llama-server batch jobs; no hardware control."""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

MAX_TOKENS = {"classify": 16, "summarize": 160, "synth": 400}
CLASSIFY_PROMPT = "Classify this customer complaint. Return exactly one of these product labels, nothing else:\n{labels}"
SUMMARIZE_PROMPT = "Summarize this customer complaint in exactly three sentences for a case handler: what happened, what the customer wants, and any amounts or dates."
SYNTH_PROMPT = "Write a realistic consumer complaint of 120 to 200 words about the product '{product}' and the issue '{issue}'. Use first person. Do not use real names, account numbers or company names; write XXXX instead."
REQUEST_DEFAULTS = {
    "temperature": 0,
    "seed": 42,
    "cache_prompt": False,
    "chat_template_kwargs": {"enable_thinking": False},
    "stream": False,
}
RETRIES = 2
RETRY_WAIT_S = 2


def read_jsonl(path):
    rows = []
    seen = set()
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
                raise ValueError(f"{path}:{line_no}: expected an object with a nonempty string id")
            if row["id"] in seen:
                raise ValueError(f"{path}:{line_no}: duplicate id {row['id']}")
            seen.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no items")
    return rows


def load_labels(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        labels = json.load(stream)["products"]
    if (not isinstance(labels, list) or len(labels) != 8
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len({label.casefold() for label in labels}) != 8):
        raise ValueError("labels.json must contain eight distinct product labels")
    return labels


def messages_for(task, item, labels=None):
    if task == "classify":
        if not labels:
            raise ValueError("Classification requires labels")
        system = CLASSIFY_PROMPT.format(labels="\n".join(labels))
        user = item["text"]
    elif task == "summarize":
        system, user = SUMMARIZE_PROMPT, item["text"]
    else:
        system = SYNTH_PROMPT.format(product=item["product"], issue=item["issue"])
        user = "Write the complaint."
    if not isinstance(user, str) or not user.strip():
        raise ValueError(f"{item['id']}: missing complaint text")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def request_completion(session, server_url, payload, timeout):
    """Retry transport, HTTP, and malformed completion failures at most twice."""
    for attempt in range(RETRIES + 1):
        try:
            response = session.post(server_url.rstrip("/") + "/v1/chat/completions",
                                    json=payload, timeout=timeout)
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            output = choice["message"]["content"]
            if not isinstance(output, str) or not output.strip():
                raise ValueError("Missing or empty completion content")
            finish_reason = choice["finish_reason"]
            if not isinstance(finish_reason, str) or not finish_reason:
                raise ValueError("Missing finish_reason")
            usage = body["usage"]
            counts = {key: usage[key] for key in ("prompt_tokens", "completion_tokens")}
            if any(type(value) is not int or value < 0 for value in counts.values()):
                raise ValueError("Invalid token usage")
            timings = body.get("timings") or {}
            reasoning = choice["message"].get("reasoning_content") or ""
            return {"output": output.strip(), "finish_reason": finish_reason, **counts,
                    "reasoning_chars": len(reasoning),
                    "prompt_ms": timings.get("prompt_ms", ""),
                    "predicted_ms": timings.get("predicted_ms", "")}
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
            if attempt == RETRIES:
                raise RuntimeError(f"Request failed after {RETRIES + 1} attempts: {exc}") from exc
            time.sleep(RETRY_WAIT_S * (attempt + 1))
    raise AssertionError("unreachable")


def wait_for_server(server_url, timeout_s):
    """Return True once GET /health answers 200 (model loaded), False after timeout_s."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            if requests.get(server_url.rstrip("/") + "/health", timeout=5).status_code == 200:
                return True
        except requests.RequestException:
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(1)


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_float(value):
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def run(args):
    source = Path(args.input)
    source_bytes = source.read_bytes()
    items = read_jsonl(source)
    if args.limit:
        items = items[:args.limit]
    labels_path = Path(args.labels) if args.labels else source.with_name("labels.json")
    labels = load_labels(labels_path) if args.task == "classify" else None
    # Validate every item before making any request.
    prompts = [{"id": item["id"], "messages": messages_for(args.task, item, labels)} for item in items]
    parameters = {**REQUEST_DEFAULTS, "max_tokens": MAX_TOKENS[args.task]}
    if args.model:
        parameters["model"] = args.model
    if not wait_for_server(args.server_url, args.wait_server):
        print(f"Error: server at {args.server_url} not ready after {args.wait_server:g} s "
              f"(GET /health did not return 200); no output written.", file=sys.stderr)
        return 1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = out_dir / f"{args.task}_{stamp}.jsonl"
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    n_failed = n_think = prompt_tokens = completion_tokens = 0
    output_hash = hashlib.sha256()
    # Exclusive creation prevents a rerun in the same second from overwriting a job.
    with output_path.open("x", encoding="utf-8", newline="\n") as stream, requests.Session() as session:
        start = time.perf_counter()
        for prompt in prompts:
            row = {"id": prompt["id"], "task": args.task, "output": "", "finish_reason": "",
                   "prompt_tokens": 0, "completion_tokens": 0, "prompt_ms": "", "predicted_ms": "",
                   "reasoning_chars": 0}
            try:
                row.update(request_completion(session, args.server_url,
                                              {**parameters, "messages": prompt["messages"]}, args.timeout))
                if "<think>" in row["output"].casefold() or row["reasoning_chars"]:
                    n_think += 1
                    row["error"] = "Completion contains thinking (<think> or reasoning_content)"
            except RuntimeError as exc:
                row["error"] = str(exc)
            if "error" in row:
                n_failed += 1
                print(f"{row['id']}: {row['error']}", file=sys.stderr)
            prompt_tokens += row["prompt_tokens"]
            completion_tokens += row["completion_tokens"]
            output_hash.update(row["output"].encode("utf-8"))
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
        elapsed = time.perf_counter() - start
    summary = {
        "task": args.task, "input_file": str(source),
        "input_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "n_items": len(items), "n_failed": n_failed, "n_think": n_think,
        "total_prompt_tokens": prompt_tokens, "total_completion_tokens": completion_tokens,
        "wall_clock_seconds": elapsed, "outputs_sha256": output_hash.hexdigest(),
        "outputs_hash_definition": "UTF-8 stripped outputs concatenated in input order, no separator",
        "request_parameters": parameters, "server_url": args.server_url,
        "endpoint": "/v1/chat/completions", "timeout_seconds": args.timeout,
        "max_retries": RETRIES, "concurrency": 1, "limit": args.limit,
        "labels_file": str(labels_path) if labels else None, "labels": labels,
        "prompt_texts": {"classify": CLASSIFY_PROMPT, "summarize": SUMMARIZE_PROMPT, "synth": SYNTH_PROMPT},
        "prompts": prompts,
        "token_accounting": "Accepted completion responses only; failed attempts may consume unreported server work",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{args.task}: {len(items)} items, {n_failed} failed, {n_think} thinking outputs")
    print(output_path)
    return 1 if n_failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=MAX_TOKENS)
    parser.add_argument("--input", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--labels")
    parser.add_argument("--model", help="Optional server model identifier, not a GGUF path")
    parser.add_argument("--timeout", type=positive_float, default=300, help="Per-attempt HTTP timeout in seconds")
    parser.add_argument("--wait-server", type=float, default=120,
                        help="Seconds to wait for GET /health to return 200 before the first request")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
