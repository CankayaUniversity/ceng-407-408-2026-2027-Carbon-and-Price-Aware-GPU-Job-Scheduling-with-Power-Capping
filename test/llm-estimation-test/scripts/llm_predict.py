"""Asks an LLM to predict the duration and energy of a GPU job (without running it).

Three information levels:
  A  zero-shot : job description and GPU name only
  B  guideline : + hardware facts, idle power, energy definitions
  C  few-shot  : + measured examples from this machine
                 (data/reference_measurements.json)

Every prediction is appended to data/raw/llm_predictions.jsonl together with
the full prompt, and is later compared with the measurement.

The prompt text and the answer schema are in Turkish on purpose: this is the
exact text used in the experiment (19 Nov 2026). Changing it would change the
experiment, so repeat runs must keep it as is.

Examples:
  python scripts/llm_predict.py --list-models                 (Gemini models)
  python scripts/llm_predict.py --model <name> --iters 30000 --mhz 1500 --levels A B C
  python scripts/llm_predict.py --model <name> --iters 10000 --mhz 0   (0 = default clock)
  python scripts/llm_predict.py --provider claude --iters 30000 --mhz 1500

Credentials: Gemini -> GEMINI_API_KEY; Claude -> ANTHROPIC_API_KEY.
"""

import argparse
from datetime import datetime, timezone
import json
import os
import time

CLAUDE_MODEL = "claude-opus-5-5"

# --- Experiment text (Turkish, kept exactly as used) -----------------------
# Answer fields: sure_s = duration (s), duvar_gucu_w = mean wall power (W),
# net_enerji_wh = net energy (Wh), gerekce = short reasoning.
SCHEMA = {
    "type": "object",
    "properties": {
        "sure_s": {"type": "number",
                   "description": "İşin tahmini süresi (saniye)"},
        "duvar_gucu_w": {"type": "number",
                         "description": "İş sırasında ortalama duvar gücü (W)"},
        "net_enerji_wh": {"type": "number",
                          "description": "Net enerji: brüt - idle*süre (Wh)"},
        "gerekce": {"type": "string",
                    "description": "Tahminin kısa gerekçesi (en fazla 3 cümle)"},
    },
    "required": ["sure_s", "duvar_gucu_w", "net_enerji_wh", "gerekce"],
    "additionalProperties": False,
}

GUIDELINE = """\
Donanım ve ölçüm bağlamı:
- Makine: laptop, NVIDIA GeForce RTX 3060 Laptop GPU (6 GB), Windows 11.
- GPU güç sınırı: varsayılan 70 W, Dynamic Boost ile ~85 W'a çıkabiliyor.
- GPU çekirdek saati varsayılan modda yük altında ~1650 MHz civarında.
  Saat kilidi verilirse (ör. 1200 MHz) GPU o frekansta sabit çalışır;
  VRAM saati değişmez.
- Ölçüm: laptopun duvardan çektiği güç (priz ölçer). Boştaki (idle) duvar
  gücü 28.1 W.
- Net enerji = iş süresince duvardan çekilen enerji - 28.1 W x süre.
- Fiziksel çerçeve: enerji = güç x süre. Hesaplama yoğun işlerde süre
  kabaca frekansla ters orantılı olur; güç frekans ve gerilimle artar.
"""


def job_description(iters, n, mhz):
  clock = ("varsayılan GPU saati (kilit yok)" if mhz == 0
           else f"GPU çekirdek saati {mhz} MHz'e kilitli (nvidia-smi -lgc)")
  return ("GPU: NVIDIA GeForce RTX 3060 Laptop GPU (laptop).\n"
          f"İş: PyTorch ile {iters} adet {n}x{n} fp16 matris çarpımı "
          f"(torch.matmul, CUDA). Ayar: {clock}.")


def few_shot(path):
  with open(path, encoding="utf-8") as f:
    rows = json.load(f)
  lines = ["Bu makinede aynı tip işin ölçülmüş sonuçları:"]
  for r in rows:
    clock = "varsayılan" if r["mhz"] == 0 else f"{r['mhz']} MHz"
    lines.append(
        f"- {r['iters']} çarpım, {r['n']}x{r['n']}, {clock}: süre "
        f"{r['duration_s']} s, ortalama duvar gücü {r['wall_w']} W, "
        f"net enerji {r['net_wh']} Wh")
  return "\n".join(lines) + "\n"


def prompt(level, iters, n, mhz, reference_path):
  parts = []
  if level in ("B", "C"):
    parts.append(GUIDELINE)
  if level == "C":
    parts.append(few_shot(reference_path))
  parts.append(job_description(iters, n, mhz))
  parts.append(
      "Bu işi YAPMA. Yalnızca bu GPU'da çalıştırılsaydı ne kadar süreceğini, "
      "ortalama duvar gücünü ve net enerjisini tahmin et. Bilgin eksikse "
      "makul varsayımlar yap ve gerekçede belirt.")
  return "\n\n".join(parts)
# ---------------------------------------------------------------------------


class RequestFailed(Exception):
  """One request failed (the other requests continue)."""


class Gemini:

  def __init__(self, model):
    from google import genai
    from google.genai import errors, types
    self.errors, self.types = errors, types
    # Key from GEMINI_API_KEY. The SDK has no timeout by default: a request
    # without an answer waited forever (hung for 48 min). The SDK's own retry
    # is off; the loop below retries and counts the attempts.
    self.client = genai.Client(http_options=types.HttpOptions(
        timeout=120_000, retry_options=types.HttpRetryOptions(attempts=1)))
    self.model = model

  def ask(self, text):
    cfg = self.types.GenerateContentConfig(
        response_mime_type="application/json", response_json_schema=SCHEMA)
    # 503 ("high demand") and 429 (rate limit) are common on the free tier;
    # they are temporary, so wait and retry. The attempt count is logged.
    for attempt in range(1, 9):
      try:
        r = self.client.models.generate_content(
            model=self.model, contents=text, config=cfg)
        break
      except self.errors.ClientError as e:
        if e.code in (401, 403):
          raise SystemExit(f"Gemini key invalid or not set: {e}")
        if e.code != 429:
          raise RequestFailed(f"client error {e.code}: {e}")
        last = e
      except self.errors.ServerError as e:
        last = e
      except self.errors.APIError as e:
        raise RequestFailed(f"API error: {e}")
      except Exception as e:  # timeout / dropped connection
        last = e
      code = getattr(last, "code", type(last).__name__)
      print(f"     {code}, attempt {attempt} failed; retrying in 30 s")
      time.sleep(30)
    else:
      raise RequestFailed(f"failed after 8 attempts: {last}")
    u = r.usage_metadata
    return {
        "model": r.model_version or self.model, "request_id": None,
        "attempt": attempt,
        "stop_reason": str(r.candidates[0].finish_reason) if r.candidates
                       else "none",
        "usage": {"input": u.prompt_token_count if u else None,
                  "output": u.candidates_token_count if u else None},
        "raw": r.text,
    }


class Claude:

  def __init__(self, model):
    import anthropic
    self.anthropic = anthropic
    self.client = anthropic.Anthropic()
    self.model = model or CLAUDE_MODEL

  def ask(self, text):
    a = self.anthropic
    try:
      r = self.client.beta.messages.create(
          model=self.model, max_tokens=16000,
          output_config={"effort": "high",
                         "format": {"type": "json_schema", "schema": SCHEMA}},
          # If a safety classifier declines, the request is handed to a
          # suitable fallback model on the server side.
          betas=["server-side-fallback-2026-07-01"], fallbacks="default",
          messages=[{"role": "user", "content": text}])
    except a.AuthenticationError:
      raise SystemExit("Anthropic API key invalid or not set.")
    except a.APIStatusError as e:
      raise RequestFailed(f"API error {e.status_code}: {e.message}")
    except a.APIConnectionError:
      raise SystemExit("Network error: API not reachable.")
    raw = (None if r.stop_reason == "refusal"
           else next(b.text for b in r.content if b.type == "text"))
    return {"model": r.model, "request_id": r._request_id,
            "stop_reason": r.stop_reason,
            "usage": {"input": r.usage.input_tokens,
                      "output": r.usage.output_tokens},
            "raw": raw}


def predict(llm, provider, level, iters, n, mhz, reference_path):
  text = prompt(level, iters, n, mhz, reference_path)
  t0 = time.monotonic()
  out = llm.ask(text)
  out["api_time_s"] = round(time.monotonic() - t0, 1)
  raw = out.pop("raw")
  try:
    prediction = json.loads(raw) if raw else None
  except ValueError:
    prediction = None
    out["parse_error"] = raw
  return {"t_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
          "provider": provider, "level": level, "iters": iters, "n": n,
          "mhz": mhz, **out, "prompt": text, "prediction": prediction}


def list_models():
  from google import genai
  client = genai.Client()  # keep a reference, or it closes while listing
  for m in client.models.list():
    if "generateContent" in (m.supported_actions or []):
      print(m.name.removeprefix("models/"), "|", m.display_name)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--provider", choices=["gemini", "claude"], default="gemini")
  p.add_argument("--model", help="Model name (required for Gemini; see "
                                 "--list-models)")
  p.add_argument("--list-models", action="store_true",
                 help="List available Gemini models and exit")
  p.add_argument("--iters", type=int)
  p.add_argument("--n", type=int, default=4096)
  p.add_argument("--mhz", type=int, help="Clock lock (MHz); 0 = default")
  p.add_argument("--levels", nargs="+", default=["A", "B", "C"],
                 choices=["A", "B", "C"])
  p.add_argument("--repeats", type=int, default=1,
                 help="How many times to ask per level (consistency)")
  p.add_argument("--reference", default="data/reference_measurements.json",
                 help="Measured examples given at level C")
  p.add_argument("--log", default="data/raw/llm_predictions.jsonl")
  a = p.parse_args()

  if a.list_models:
    list_models()
    return
  if a.iters is None or a.mhz is None:
    p.error("--iters and --mhz are required")
  if a.provider == "gemini":
    if not a.model:
      p.error("--model is required for Gemini (see --list-models)")
    llm = Gemini(a.model)
  else:
    llm = Claude(a.model)

  os.makedirs(os.path.dirname(a.log), exist_ok=True)
  with open(a.log, "a", encoding="utf-8") as f:
    for level in a.levels:
      for _ in range(a.repeats):
        try:
          rec = predict(llm, a.provider, level, a.iters, a.n, a.mhz,
                        a.reference)
        except RequestFailed as e:
          print(f"[{level}] {e}")
          continue
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        time.sleep(5)  # stay under the free tier's per-minute limit
        t = rec["prediction"] or {}
        print(f"[{level}] duration {t.get('sure_s')} s | wall "
              f"{t.get('duvar_gucu_w')} W | net {t.get('net_enerji_wh')} Wh | "
              f"{rec['stop_reason']}")
        if t:
          print(f"     reasoning: {t['gerekce']}")


if __name__ == "__main__":
  main()
