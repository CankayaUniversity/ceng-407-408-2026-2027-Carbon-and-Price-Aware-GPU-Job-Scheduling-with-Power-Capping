"""CLI integration tests using canned responses; no llama-server or GPU."""

import hashlib
import json
import subprocess
import sys
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import unittest
import tempfile
from contextlib import contextmanager
from math import isclose

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from evaluate_outputs import classification_metrics, parse_label, sentence_count
from run_tasks import MAX_TOKENS, SUMMARIZE_PROMPT, SYNTH_PROMPT

LABELS = ["Debt collection", "Checking or savings account", "Credit card",
          "Money transfer, virtual currency, or money service", "Mortgage", "Vehicle loan or lease",
          "Credit reporting or other personal consumer reports", "Student loan"]


def completion(text="Credit card", timings=True):
    body = {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 3}}
    if timings:
        body["timings"] = {"prompt_ms": 1.25, "predicted_ms": 2.5}
    return body


@contextmanager
def fake_server():
    pending, received = deque(), []
    health = {"ready": True}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = b'{"status": "ok"}' if self.path == "/health" and health["ready"] else b'{"error": "loading"}'
            self.send_response(200 if self.path == "/health" and health["ready"] else 503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            received.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            status, body = pending.popleft() if pending else (500, {"error": "no canned response"})
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", pending, received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def task_input(tmp_path):
    (tmp_path / "labels.json").write_text(json.dumps({"products": LABELS}), encoding="utf-8")
    return write_jsonl(tmp_path / "input.jsonl", [
        {"id": "a", "text": "I disputed a card charge.", "product": "Credit card", "issue": "Fees"},
        {"id": "b", "text": "My mortgage payment is missing.", "product": "Mortgage", "issue": "Payment"},
    ])


def cli(script, *args):
    return subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)],
                          capture_output=True, text=True, timeout=20)


def runner(task_input, tmp_path, server_url, task="classify", *extra):
    return cli("run_tasks.py", "--task", task, "--input", task_input,
               "--server-url", server_url, "--out-dir", tmp_path / "outputs", *extra)


def saved(tmp_path):
    output = next((tmp_path / "outputs").glob("*.jsonl"))
    return [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()], json.loads(
        output.with_name(output.stem + "_summary.json").read_text(encoding="utf-8"))


def check_payload_and_summary(fake_server, task_input, tmp_path, task):
    url, pending, received = fake_server
    pending.extend([(200, completion()), (200, completion("Mortgage", timings=False))])
    result = runner(task_input, tmp_path, url + "/", task)
    assert result.returncode == 0, result.stderr
    assert len(received) == 2
    for endpoint, payload in received:
        assert endpoint == "/v1/chat/completions"
        assert payload["temperature"] == 0
        assert payload["seed"] == 42
        assert payload["cache_prompt"] is False
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["max_tokens"] == MAX_TOKENS[task]
        assert payload["stream"] is False
    messages = received[0][1]["messages"]
    if task == "classify":
        assert all(label in messages[0]["content"] for label in LABELS)
        assert messages[1]["content"] == "I disputed a card charge."
    elif task == "summarize":
        assert messages[0]["content"] == SUMMARIZE_PROMPT
    else:
        assert messages[0]["content"] == SYNTH_PROMPT.format(product="Credit card", issue="Fees")
    rows, summary = saved(tmp_path)
    assert [row["id"] for row in rows] == ["a", "b"]
    assert rows[0]["prompt_ms"] == 1.25 and rows[1]["prompt_ms"] == ""
    assert summary["n_items"] == 2 and summary["n_failed"] == summary["n_think"] == 0
    assert summary["input_sha256"] == hashlib.sha256(task_input.read_bytes()).hexdigest()
    assert summary["outputs_sha256"] == hashlib.sha256(b"Credit cardMortgage").hexdigest()
    assert summary["total_prompt_tokens"] == 40 and summary["total_completion_tokens"] == 6
    assert summary["wall_clock_seconds"] >= 0
    assert summary["prompts"][0]["messages"] == messages


def check_retry_then_success(fake_server, task_input, tmp_path):
    url, pending, received = fake_server
    pending.extend([(503, {}), (200, {"choices": []}), (200, completion())])
    result = runner(task_input, tmp_path, url, "classify", "--limit", 1)
    assert result.returncode == 0, result.stderr
    assert len(received) == 3
    assert all(payload == received[0][1] for _, payload in received)


def check_permanent_failure_is_recorded_and_continues(fake_server, task_input, tmp_path):
    url, pending, received = fake_server
    pending.extend([(500, {})] * 3 + [(200, completion("Mortgage"))])
    result = runner(task_input, tmp_path, url)
    assert result.returncode != 0
    assert len(received) == 4
    rows, summary = saved(tmp_path)
    assert len(rows) == 2 and rows[0]["error"] and rows[0]["output"] == ""
    assert rows[1]["output"] == "Mortgage" and summary["n_failed"] == 1


def check_thinking_fails_without_retry(fake_server, task_input, tmp_path):
    url, pending, received = fake_server
    pending.append((200, completion("<think>reason</think>Credit card")))
    result = runner(task_input, tmp_path, url, "classify", "--limit", 1)
    assert result.returncode != 0
    rows, summary = saved(tmp_path)
    assert len(received) == 1
    assert summary["n_failed"] == summary["n_think"] == 1
    assert "<think>" in rows[0]["output"]


def check_reasoning_content_fails(fake_server, task_input, tmp_path):
    url, pending, received = fake_server
    body = completion("Credit card")
    body["choices"][0]["message"]["reasoning_content"] = "hidden reasoning"
    pending.append((200, body))
    result = runner(task_input, tmp_path, url, "classify", "--limit", 1)
    assert result.returncode != 0
    rows, summary = saved(tmp_path)
    assert len(received) == 1
    assert summary["n_failed"] == summary["n_think"] == 1
    assert rows[0]["reasoning_chars"] == len("hidden reasoning")


def check_server_not_ready(task_input, tmp_path):
    result = runner(task_input, tmp_path / "down", "http://127.0.0.1:9", "classify", "--wait-server", 1)
    assert result.returncode != 0
    assert "not ready" in result.stderr
    assert not (tmp_path / "down" / "outputs").exists()


def check_label_parsing(text):
    assert parse_label(text, LABELS) == "Credit card"


def check_unparseable_label(text):
    assert parse_label(text, LABELS) is None


def check_classification_metrics_penalize_unparseable():
    metrics = classification_metrics(["Credit card", "Credit card", "Mortgage"],
                                     ["Credit card", None, "Credit card"], LABELS)
    assert isclose(metrics["accuracy"], 1 / 3)
    assert isclose(metrics["macro_f1"], 0.5 / 8)
    assert metrics["n_unparseable"] == 1
    assert metrics["confusion_matrix"]["Credit card"]["<unparseable>"] == 1


def check_evaluate_classification_cli(task_input, tmp_path):
    output = write_jsonl(tmp_path / "classify.jsonl", [
        {"id": "b", "task": "classify", "output": "Mortgage"},
        {"id": "a", "task": "classify", "output": '"Credit card."'},
    ])
    result = cli("evaluate_outputs.py", output, "--input", task_input)
    assert result.returncode == 0, result.stderr
    report = json.loads(Path(str(output) + "_eval.json").read_text())
    assert report["accuracy"] == 1 and report["n_missing"] == 0


def check_evaluate_text_and_reclassify(fake_server, task_input, tmp_path, task):
    url, pending, received = fake_server
    text = "I paid $12.50. I want a refund. This happened yesterday."
    output = write_jsonl(tmp_path / "text.jsonl", [
        {"id": "a", "task": task, "output": text},
        {"id": "b", "task": task, "output": text},
    ])
    pending.extend([(200, completion()), (200, completion("Mortgage"))])
    result = cli("evaluate_outputs.py", output, "--input", task_input, "--reclassify", "--server-url", url)
    assert result.returncode == 0, result.stderr
    report = json.loads(Path(str(output) + "_eval.json").read_text())
    assert report["reclassification"]["accuracy"] == 1
    assert received[0][1]["messages"][1]["content"] == text
    if task == "summarize":
        assert report["sentence_count"]["counts"] == {"3": 2}
    else:
        assert report["exact_duplicate_groups"] == [["a", "b"]]
        assert report["near_duplicate_groups"] == [["a", "b"]]


def check_compare_cli(tmp_path, variant):
    left = [{"id": "a", "output": "alpha"}, {"id": "b", "output": "beta"}]
    right = list(reversed([dict(row) for row in left]))
    if variant == "different":
        right[0]["output"] = "changed"
    elif variant == "missing":
        right.pop()
    elif variant == "duplicate":
        right.append(right[0])
    elif variant == "failed":
        right[0]["error"] = "request failed"
    a = write_jsonl(tmp_path / "a.jsonl", left)
    b = write_jsonl(tmp_path / "b.jsonl", right)
    result = cli("compare_outputs.py", a, b)
    assert (result.returncode == 0) == (variant == "same")


def check_invalid_input_before_network(fake_server, task_input, tmp_path):
    url, pending, received = fake_server
    write_jsonl(task_input, [{"id": "a", "text": "ok"}, {"id": "b"}])
    assert runner(task_input, tmp_path, url).returncode != 0
    assert not received


def check_sentence_count():
    assert sentence_count("Paid $12.50. Need it back! Please help?") == 3


class TaskToolsTests(unittest.TestCase):
    """Runs with unittest or pytest; each network test owns its fake server."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.input = task_input(self.path)

    def test_payloads(self):
        for task in ("classify", "summarize", "synth"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as folder, fake_server() as server:
                check_payload_and_summary(server, self.input, Path(folder), task)

    def test_retry(self):
        with fake_server() as server:
            check_retry_then_success(server, self.input, self.path)

    def test_failure(self):
        with fake_server() as server:
            check_permanent_failure_is_recorded_and_continues(server, self.input, self.path)

    def test_thinking(self):
        with fake_server() as server:
            check_thinking_fails_without_retry(server, self.input, self.path)

    def test_server_not_ready(self):
        check_server_not_ready(self.input, self.path)

    def test_reasoning_content(self):
        with fake_server() as server:
            check_reasoning_content_fails(server, self.input, self.path)

    def test_label_parsing(self):
        for text in ('"Credit card."', " credit CARD. ", "'Credit card'.", "“Credit card.”"):
            with self.subTest(text=text):
                check_label_parsing(text)
        for text in ("It is Credit card", "Credit card or Mortgage", "", "Credit card\nExplanation"):
            with self.subTest(text=text):
                check_unparseable_label(text)

    def test_metrics(self):
        check_classification_metrics_penalize_unparseable()

    def test_evaluate_classification(self):
        check_evaluate_classification_cli(self.input, self.path)

    def test_evaluate_text(self):
        for task in ("summarize", "synth"):
            with self.subTest(task=task), fake_server() as server:
                check_evaluate_text_and_reclassify(server, self.input, self.path, task)

    def test_compare(self):
        for variant in ("same", "different", "missing", "duplicate", "failed"):
            with self.subTest(variant=variant):
                check_compare_cli(self.path, variant)

    def test_invalid_input(self):
        with fake_server() as server:
            check_invalid_input_before_network(server, self.input, self.path)

    def test_sentence_count(self):
        check_sentence_count()


if __name__ == "__main__":
    unittest.main()
