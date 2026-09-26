"""
blocking.py

Candidate generation for the Business Entity Resolution Challenge.

Three blocking layers, unioned together:
  1. Exact-key buckets on normalize_name().sorted_key and .prefix_key
     -- fast, catches suffix variation / word reordering.
  2. MinHash LSH over character 3-grams of the cleaned name
     -- catches typos (e.g. "Wilblims" vs "Williams") that exact keys miss.
  3. Address-token buckets (first significant address token, when address
     is present) -- catches cases where the name diverges but the
     address doesn't (e.g. the "Drexkor" example in the sample data).

Usage:
    pip install datasketch pyarrow tqdm

    # quick smoke test on a subset while you're developing
    python blocking.py --sample 5000

    # full run against train (also reports recall vs ground truth)
    python blocking.py

    # full run against test (no ground truth to check against)
    python blocking.py --split test
"""

import argparse
import re
from collections import defaultdict

import pandas as pd
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

from normalize import normalize_address, normalize_name

NUM_PERM = 64          # MinHash permutations -- higher = more accurate, slower
LSH_THRESHOLD = 0.35    # Jaccard similarity threshold for the LSH index
SHINGLE_SIZE = 3        # character n-gram size for MinHash


def char_shingles(text: str, n: int = SHINGLE_SIZE):
    text = text.replace(" ", "")
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def build_minhash(text: str) -> MinHash:
    mh = MinHash(num_perm=NUM_PERM)
    for shingle in char_shingles(text):
        mh.update(shingle.encode("utf8"))
    return mh


_DIGIT_RUN_RE = re.compile(r"\d{4,}")


def address_block_token(norm_addr) -> str:
    """A single coarse blocking token from an address: prefer a postal-code-like
    digit run (4+ digits), else the first non-trivial alphabetic token."""
    if norm_addr.is_missing:
        return ""
    m = _DIGIT_RUN_RE.search(norm_addr.cleaned)
    if m:
        return m.group(0)
    for tok in norm_addr.tokens:
        if len(tok) >= 4:
            return tok
    return ""


def load_and_normalize(path: str, sample: int | None) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if sample:
        df = df.head(sample).copy()
    print(f"  normalizing {len(df):,} rows from {path} ...")
    names = [normalize_name(n) for n in tqdm(df["business_name"], desc="names")]
    addrs = [normalize_address(a) for a in tqdm(df["business_address"], desc="addresses")]
    df["name_cleaned"] = [n.cleaned for n in names]
    df["name_sorted_key"] = [n.sorted_key for n in names]
    df["name_prefix_key"] = [n.prefix_key for n in names]
    df["addr_block_token"] = [address_block_token(a) for a in addrs]
    return df


def build_exact_key_index(df: pd.DataFrame, key_col: str) -> dict:
    idx = defaultdict(list)
    for eid, key in zip(df["entity_id"], df[key_col]):
        if key:
            idx[key].append(eid)
    return idx


def build_lsh_index(df: pd.DataFrame) -> tuple[MinHashLSH, dict]:
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    minhashes = {}
    for eid, cleaned in tqdm(
        zip(df["entity_id"], df["name_cleaned"]), total=len(df), desc="building LSH index"
    ):
        mh = build_minhash(cleaned)
        minhashes[eid] = mh
        lsh.insert(eid, mh)
    return lsh, minhashes


def build_address_index(df: pd.DataFrame) -> dict:
    idx = defaultdict(list)
    for eid, token in zip(df["entity_id"], df["addr_block_token"]):
        if token:
            idx[token].append(eid)
    return idx


def generate_candidates(
    s1: pd.DataFrame,
    other_indexes: list[tuple[str, dict, dict, MinHashLSH]],
) -> dict:
    """other_indexes: list of (source_name, sorted_key_idx, prefix_key_idx, addr_idx, lsh, minhash_lookup)
    Returns {s1_entity_id: set(candidate_ids)}"""
    candidates = defaultdict(set)

    for _, row in tqdm(s1.iterrows(), total=len(s1), desc="generating candidates"):
        eid = row["entity_id"]
        sk, pk, at = row["name_sorted_key"], row["name_prefix_key"], row["addr_block_token"]
        mh = build_minhash(row["name_cleaned"])

        for _, sorted_idx, prefix_idx, addr_idx, lsh in other_indexes:
            if sk:
                candidates[eid].update(sorted_idx.get(sk, []))
            if pk:
                candidates[eid].update(prefix_idx.get(pk, []))
            if at:
                candidates[eid].update(addr_idx.get(at, []))
            candidates[eid].update(lsh.query(mh))

    return candidates


def measure_recall(candidates: dict, ground_truth: pd.DataFrame) -> None:
    total_true = 0
    found_true = 0
    total_entities = 0
    total_candidates = 0
    perfect_singletons = 0
    total_singletons = 0

    for _, row in ground_truth.iterrows():
        eid = row["source1_entity_id"]
        raw = row["matched_entity_ids"]
        true_ids = set() if pd.isna(raw) or str(raw).strip() == "" else set(str(raw).split(","))
        cand = candidates.get(eid, set())

        total_entities += 1
        total_candidates += len(cand)

        if not true_ids:
            total_singletons += 1
            # a singleton is "recoverable" if we generated few/no false candidates,
            # but that's a precision question for the model stage, not blocking recall
            continue

        total_true += len(true_ids)
        found_true += len(true_ids & cand)

    recall = found_true / total_true if total_true else float("nan")
    avg_candidates = total_candidates / total_entities if total_entities else 0

    print(f"\n=== BLOCKING RECALL REPORT ===")
    print(f"Entities checked:      {total_entities:,}")
    print(f"Singletons:            {total_singletons:,}")
    print(f"True matches total:    {total_true:,}")
    print(f"True matches found:    {found_true:,}")
    print(f"RECALL CEILING:        {recall:.4f}")
    print(f"Avg candidates per S1: {avg_candidates:.1f}")


def write_candidates(candidates: dict, out_path: str, s1_ids) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid in s1_ids:
            cand = candidates.get(eid, set())
            f.write(f"{eid}\t{','.join(sorted(cand))}\n")
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--sample", type=int, default=None, help="limit rows for a quick test run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = f"dataset/{args.split}"
    out_path = args.out or f"output/candidate_pairs_{args.split}.tsv"

    print("Loading + normalizing source1 ...")
    s1 = load_and_normalize(f"{base}/{args.split}_source1.tsv", args.sample)
    print("Loading + normalizing source2 ...")
    s2 = load_and_normalize(f"{base}/{args.split}_source2.tsv", args.sample)
    print("Loading + normalizing source3 ...")
    s3 = load_and_normalize(f"{base}/{args.split}_source3.tsv", args.sample)

    other_indexes = []
    for name, df in [("source2", s2), ("source3", s3)]:
        print(f"Building indexes for {name} ...")
        sorted_idx = build_exact_key_index(df, "name_sorted_key")
        prefix_idx = build_exact_key_index(df, "name_prefix_key")
        addr_idx = build_address_index(df)
        lsh, _ = build_lsh_index(df)
        other_indexes.append((name, sorted_idx, prefix_idx, addr_idx, lsh))

    print("Generating candidates for source1 entities ...")
    candidates = generate_candidates(s1, other_indexes)

    import os
    os.makedirs("output", exist_ok=True)
    write_candidates(candidates, out_path, s1["entity_id"])

    if args.split == "train":
        gt_path = f"{base}/train_ground_truth.tsv"
        gt = pd.read_csv(gt_path, sep="\t")
        if args.sample:
            gt = gt[gt["source1_entity_id"].isin(s1["entity_id"])]
        measure_recall(candidates, gt)


if __name__ == "__main__":
    main()
