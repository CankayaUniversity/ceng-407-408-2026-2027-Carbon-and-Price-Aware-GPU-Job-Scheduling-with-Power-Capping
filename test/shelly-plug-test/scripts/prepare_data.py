import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

COLUMNS = {
    "id": "Complaint ID",
    "date": "Date received",
    "product": "Product",
    "sub_product": "Sub-product",
    "issue": "Issue",
    "sub_issue": "Sub-issue",
    "narrative": "Consumer complaint narrative",
    "company": "Company",
}

CLASSIFY_WORDS = (50, 270)
SUMMARIZE_WORDS = (270, 1000)
SYNTH_PER_PRODUCT = 4
TR_COUNT = 50
RESERVOIR_SIZE = 3000

SOURCE_URL = ("https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/"
              "cfpb-consumer-complaint-database-narratives-archive/")


def find_columns(header):
    missing = [name for name in COLUMNS.values() if name not in header]
    if missing:
        print("These columns are missing from the file:", missing)
        sys.exit(1)
    return {key: header.index(name) for key, name in COLUMNS.items()}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def open_csv(path):
    if path.lower().endswith(".zip"):
        archive = zipfile.ZipFile(path)
        csv_names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            print("No CSV inside the ZIP. Contents:")
            for n in archive.namelist():
                print("  ", n)
            sys.exit(1)
        if len(csv_names) > 1:
            print("More than one CSV in the ZIP, using the largest:", csv_names)
            csv_names.sort(key=lambda n: archive.getinfo(n).file_size, reverse=True)
        stream = io.TextIOWrapper(archive.open(csv_names[0]), encoding="utf-8-sig", errors="replace", newline="")
        return csv_names[0], stream
    return os.path.basename(path), open(path, encoding="utf-8-sig", errors="replace", newline="")


def clean_text(text):
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def word_stats(rows):
    counts = sorted(r["n_words"] for r in rows)
    if not counts:
        return {}
    return {"n": len(counts), "min": counts[0], "median": counts[len(counts) // 2], "max": counts[-1]}


PERCENTILES = (10, 25, 50, 75, 90, 95, 99)


def percentile(sorted_values, p):
    index = round(p / 100 * (len(sorted_values) - 1))
    return sorted_values[index]


def print_length_table(word_lengths):
    print("\nWords per narrative (percentiles):")
    header = "".join(f"{'p' + str(p):>7}" for p in PERCENTILES)
    print(f"   {'product':<52}{'n':>7}{header}{'max':>8}")
    rows = sorted(word_lengths.items(), key=lambda item: -len(item[1]))
    rows.append(("ALL", [n for values in word_lengths.values() for n in values]))
    for product, values in rows:
        values = sorted(values)
        cells = "".join(f"{percentile(values, p):>7}" for p in PERCENTILES)
        print(f"   {product[:51]:<52}{len(values):>7}{cells}{values[-1]:>8}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, help="ZIP (or CSV) downloaded from the CFPB FOIA archive")
    parser.add_argument("--out", default="data", help="output folder")
    parser.add_argument("--products", type=int, default=8, help="number of products (classes)")
    parser.add_argument("--classify", type=int, default=60, help="classification test complaints per product")
    parser.add_argument("--classify-dev", type=int, default=10, help="classification dev complaints per product")
    parser.add_argument("--summarize", type=int, default=120, help="summarization test complaints (total)")
    parser.add_argument("--summarize-dev", type=int, default=10, help="summarization dev complaints (total)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect-only", action="store_true", help="only print columns and counts, write nothing")
    args = parser.parse_args()

    if not os.path.exists(args.zip):
        print("File not found:", args.zip)
        return
    print(f"Source: {args.zip} ({os.path.getsize(args.zip) / 1e6:.0f} MB)")
    source_hash = sha256(args.zip)
    inner_name, stream = open_csv(args.zip)
    reader = csv.reader(stream)
    header = next(reader)
    cols = find_columns(header)
    print("CSV:", inner_name)
    print("Columns:", header)

    rng = random.Random(args.seed)
    seen = Counter()
    pool = defaultdict(list)
    product_counts = Counter()
    issue_counts = defaultdict(Counter)
    fingerprints = set()
    word_lengths = defaultdict(list)
    total_rows = with_narrative = duplicates = 0

    def get(row, key):
        i = cols[key]
        return row[i].strip() if i < len(row) else ""

    for row in reader:
        total_rows += 1
        if total_rows % 200000 == 0:
            print(f"   {total_rows:,} rows read, {with_narrative:,} with narrative")
        text = get(row, "narrative")
        if not text:
            continue
        with_narrative += 1
        product = get(row, "product")
        product_counts[product] += 1
        issue_counts[product][get(row, "issue")] += 1
        if args.inspect_only:
            word_lengths[product].append(len(clean_text(text).split()))
            continue
        text = clean_text(text)
        fingerprint = hashlib.md5(re.sub(r"\W+", "", text.lower())[:400].encode()).hexdigest()
        if fingerprint in fingerprints:
            duplicates += 1
            continue
        fingerprints.add(fingerprint)
        n_words = len(text.split())
        if CLASSIFY_WORDS[0] <= n_words < CLASSIFY_WORDS[1]:
            bucket = "classify"
        elif SUMMARIZE_WORDS[0] <= n_words <= SUMMARIZE_WORDS[1]:
            bucket = "summarize"
        else:
            continue
        record = {
            "id": get(row, "id"), "date_received": get(row, "date"), "product": product,
            "sub_product": get(row, "sub_product"), "issue": get(row, "issue"),
            "sub_issue": get(row, "sub_issue"), "company": get(row, "company"),
            "n_words": n_words, "n_chars": len(text), "text": text,
        }
        key = (product, bucket)
        seen[key] += 1
        if len(pool[key]) < RESERVOIR_SIZE:
            pool[key].append(record)
        else:
            j = rng.randrange(seen[key])
            if j < RESERVOIR_SIZE:
                pool[key][j] = record

    print(f"\nRows: {total_rows:,}   with narrative: {with_narrative:,}   duplicate narratives dropped: {duplicates:,}")
    print("\nNarratives per product:")
    for product, n in product_counts.most_common():
        print(f"   {n:>8,}  {product}")
    if args.inspect_only:
        print_length_table(word_lengths)
        return

    needed = args.classify + args.classify_dev
    eligible = [p for p, _ in product_counts.most_common() if len(pool[(p, "classify")]) >= needed]
    products = eligible[: args.products]
    if len(products) < args.products:
        print(f"\nWarning: only {len(products)} products have at least {needed} suitable complaints.")
    print("\nClassification labels:", products)

    for key in pool:
        pool[key].sort(key=lambda r: r["id"])
    rng = random.Random(args.seed)

    classify_test, classify_dev = [], []
    for product in products:
        chosen = rng.sample(pool[(product, "classify")], needed)
        classify_dev += chosen[: args.classify_dev]
        classify_test += chosen[args.classify_dev:]
    rng.shuffle(classify_test)
    rng.shuffle(classify_dev)

    summarize_total = args.summarize + args.summarize_dev
    quotas = {p: summarize_total // len(products) + (1 if i < summarize_total % len(products) else 0)
              for i, p in enumerate(products)}
    summarize_all, shortfall = [], 0
    for product in products:
        candidates = pool[(product, "summarize")]
        n = min(quotas[product], len(candidates))
        shortfall += quotas[product] - n
        summarize_all += rng.sample(candidates, n)
    if shortfall:
        remaining = [r for p in products for r in pool[(p, "summarize")] if r not in summarize_all]
        summarize_all += rng.sample(remaining, min(shortfall, len(remaining)))
    rng.shuffle(summarize_all)
    summarize_dev = summarize_all[: args.summarize_dev]
    summarize_test = summarize_all[args.summarize_dev:]

    synth_spec = []
    for product in products:
        for issue, _ in issue_counts[product].most_common(SYNTH_PER_PRODUCT):
            synth_spec.append({"id": f"synth_{len(synth_spec) + 1:03d}", "product": product, "issue": issue})

    tr_sample = sorted(rng.sample(classify_test, min(TR_COUNT, len(classify_test))), key=lambda r: r["id"])

    os.makedirs(args.out, exist_ok=True)
    outputs = {
        "classify_test.jsonl": classify_test,
        "classify_dev.jsonl": classify_dev,
        "summarize_test.jsonl": summarize_test,
        "summarize_dev.jsonl": summarize_dev,
        "synth_spec.jsonl": synth_spec,
        "tr_subset.jsonl": [{"id": r["id"], "product": r["product"], "text_en": r["text"]} for r in tr_sample],
    }
    for name, rows in outputs.items():
        write_jsonl(os.path.join(args.out, name), rows)

    labels = {
        "products": products,
        "issues_by_product": {p: [i for i, _ in issue_counts[p].most_common()] for p in products},
        "narratives_per_product_in_source": {p: product_counts[p] for p in products},
    }
    with open(os.path.join(args.out, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "CFPB Consumer Complaint Database narratives archive (FOIA Electronic Reading Room)",
        "source_url": SOURCE_URL,
        "source_file": os.path.basename(args.zip),
        "inner_csv": inner_name,
        "source_sha256": source_hash,
        "rows_total": total_rows,
        "rows_with_narrative": with_narrative,
        "duplicates_dropped": duplicates,
        "seed": args.seed,
        "params": {
            "products": args.products,
            "classify_per_product": args.classify,
            "classify_dev_per_product": args.classify_dev,
            "summarize": args.summarize,
            "summarize_dev": args.summarize_dev,
            "classify_words": CLASSIFY_WORDS,
            "summarize_words": SUMMARIZE_WORDS,
        },
        "files": {name: {"rows": len(rows), "sha256": sha256(os.path.join(args.out, name))}
                  for name, rows in outputs.items()},
        "word_stats": {"classify_test": word_stats(classify_test), "summarize_test": word_stats(summarize_test)},
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to '{args.out}':")
    for name, rows in outputs.items():
        print(f"   {name:<22} {len(rows):>4} rows")
    print("   labels.json, manifest.json")
    stats = manifest["word_stats"]
    print(f"\nClassification words: median {stats['classify_test'].get('median')}, "
          f"summarization words: median {stats['summarize_test'].get('median')}")


if __name__ == "__main__":
    main()