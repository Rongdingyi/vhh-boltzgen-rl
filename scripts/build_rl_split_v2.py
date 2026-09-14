#!/usr/bin/env python
"""Build a clean RL train/held-out split from SAbDab2 crystal VHH structures.

Pipeline (single CPU Slurm job):
 1. Extract VHH-like chains (110-135 aa, J-motif) + antigen chain info from
    /share/data/limc/structure_lm_data/Sabdab2.
 2. MMseqs2 70% easy-cluster.
 3. Leakage check vs the round-1 valid100 manifest (mmseqs easy-search).
 4. Cluster-level split: 24 train + 8 held-out clusters (seed fixed).
 5. ANARCII (vnar, IMGT) numbering -> CDR1/2/3 0-based positions.
 6. Package per-case BoltzGen IF inputs (backbone.cif + intermediate_designs
    design.cif/design.npz + design.yaml) exactly as the round-1 pipeline
    consumes them (see vhh_rl/cli/common.py::capture_cases).
 7. Write manifest + summary.

Outputs under runs/round1_rl_split/.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import gemmi
import numpy as np

SABDAB2 = Path("/share/data/limc/structure_lm_data/Sabdab2")
MANIFEST100 = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl/runs/round1/rl_manifest.jsonl")
OUT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl/runs/round1_rl_split")
PY = sys.executable
SEED = 20260911
N_TRAIN, N_HELD = 24, 8
CDR_IMGT = {"cdr1": range(27, 39), "cdr2": range(56, 66), "cdr3": range(105, 118)}
J_MOTIF = "WG.G"  # conserved J-region start (WGQG/WGRG), filters non-Ig chains
MMSEQS = "/share/app/mmseqs2/bin/mmseqs"

def one_letter(chain) -> str:
    s = gemmi.one_letter_code([r.name for r in chain if r.entity_type in (gemmi.EntityType.Polymer, gemmi.EntityType.Branched, gemmi.EntityType.Unknown)])
    return s.upper().replace("X", "")


def extract() -> list[dict]:
    """Collect VHH-like chains and antigen info from every cif."""
    cands = []
    for f in sorted(os.listdir(SABDAB2)):
        if not f.endswith(".cif"):
            continue
        stem = f[4:-4]
        try:
            txt = bytes.fromhex(stem).decode()
        except Exception:
            continue
        parts = txt.split("_")
        pdbid = parts[1][4:] if len(parts) >= 4 and parts[1].startswith("0000") else (parts[1] if len(parts) >= 4 else "?")
        try:
            st = gemmi.read_structure(str(SABDAB2 / f))
            st.setup_entities()
            st.remove_hydrogens()
            st.remove_waters()
            st.remove_alternative_conformations()
            model = st[0]
        except Exception as e:
            print(f"  parse fail {f}: {e}")
            continue
        chains = [(ch.name, one_letter(ch)) for ch in model]
        vhhs = [(n, s) for n, s in chains if 110 <= len(s) <= 135]
        if not vhhs:
            continue
        # keep only chains that look like VHH (J-region + no light chain pairing)
        import re
        for cname, seq in vhhs:
            if not re.search(J_MOTIF, seq):
                continue
            # light chain partner? (classic antibody chains ~107-115 with CL motif)
            partners = [(n, s) for n, s in chains if n != cname and len(s) >= 40]
            has_light = any(re.search(r"GQPKAAP|GQPKAN|TVAAPSV", s) for _, s in partners)
            if has_light:
                continue
            antigens = [(n, len(s)) for n, s in partners if len(s) >= 50]
            cands.append(
                dict(pdbid=pdbid, chain=cname, seq=seq, file=str(SABDAB2 / f),
                     n_antigens=len(antigens), antigen_chains=[n for n, _ in antigens],
                     single=True)
            )
    # dedupe identical (pdbid, chain)
    seen, out = set(), []
    for c in cands:
        k = (c["pdbid"], c["chain"])
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out


def cluster(fasta: Path, tmp: Path) -> dict[str, str]:
    """mmseqs easy-cluster at 70% identity; returns seq_header -> rep_header."""
    tmp.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [MMSEQS, "easy-cluster", str(fasta), str(tmp / "clu"), str(tmp / "tmp"),
         "--min-seq-id", "0.7", "-c", "0.8", "--cov-mode", "0",
         "--threads", str(os.cpu_count() or 8)],
        check=True, capture_output=True)
    rep = {}
    with open(tmp / "clu_cluster.tsv") as fh:
        for line in fh:
            r, m = line.split()
            rep[m] = r
    return rep


def valid100_seqs() -> dict[str, str]:
    seqs = {}
    for line in open(MANIFEST100):
        r = json.loads(line)
        seqs[r["case_id"]] = r["full_sequence"].upper()
    return seqs


def leakage_check(valid: dict[str, str], fasta: Path, tmp: Path) -> set[str]:
    """Candidate headers with >=90% identity to any valid100 native sequence."""
    vfast = tmp / "valid100.fasta"
    with open(vfast, "w") as fh:
        for cid, s in valid.items():
            fh.write(f">{cid}\n{s}\n")
    subprocess.run(
        [MMSEQS, "easy-search", str(vfast), str(fasta), str(tmp / "leak.tsv"),
         str(tmp / "leak_tmp"), "--search-type", "1",
         "--threads", str(os.cpu_count() or 8)],
        check=True, capture_output=True)
    hit = set()
    for line in open(tmp / "leak.tsv"):
        f = line.split("\t")
        q, t, pid = f[0], f[1], float(f[2])
        if pid >= 0.85:
            hit.add(t)
    return hit


def anarcii_number(seqs: list[tuple[str, str]]) -> dict[str, dict]:
    from anarcii import Anarcii
    model = Anarcii(seq_type="vnar", mode="speed",
                    batch_size=8, cpu=True, ncpu=4, verbose=False)
    out = model.number(dict(seqs))
    if isinstance(out, tuple):
        out = out[0]
    res = {}
    for name, item in (out.items() if isinstance(out, dict) else
                       [(x.get("query_name", "?"), x) for x in out]):
        res[name] = item
    return res


def cdr_positions(item: dict, seq: str) -> tuple[dict[str, list[int]], list[int]] | None:
    """Map anarcii IMGT numbering to 0-based sequence indices.

    '-' entries are IMGT-grid placeholders (residue absent from the chain) and
    consume no sequence index. Returns None if letters cannot reconstruct seq.
    """
    numbering = item["numbering"]
    letters = []
    for entry in numbering:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            return None
        res = entry[1]
        if res != "-":
            letters.append(res)
    if "".join(letters) != seq:
        return None
    pos = {"cdr1": [], "cdr2": [], "cdr3": []}
    fr = []
    i = 0
    for entry in numbering:
        res = entry[1]
        if res == "-":
            continue
        num = entry[0]
        n = int(num[0]) if isinstance(num, (list, tuple)) and num else None
        if n is None:
            return None
        for name, rng in CDR_IMGT.items():
            if n in rng:
                pos[name].append(i)
                break
        else:
            fr.append(i)
        i += 1
    return pos, fr


def write_cif(src: str, out: Path, chain_name: str) -> str:
    st = gemmi.read_structure(src)
    st.setup_entities()
    st.remove_hydrogens()
    st.remove_waters()
    st.remove_alternative_conformations()
    st.setup_entities()
    model = st[0]
    keep = model[chain_name]
    new = gemmi.Structure()
    m = gemmi.Model("1")
    ch = gemmi.Chain("A")
    for r in keep:
        ch.add_residue(r)
    m.add_chain(ch)
    new.add_model(m)
    new.name = st.name
    new.setup_entities()
    new.assign_label_seq_id()
    doc = new.make_mmcif_document()
    doc.write_file(str(out))
    return chain_name


def write_spec(seq: str, cdr: dict[str, list[int]], out: Path):
    """Mask-token spec in the same digit format as the round-1 design.yaml."""
    masked = list(seq)
    blocks = [cdr["cdr1"], cdr["cdr2"], cdr["cdr3"]]
    for b in blocks:
        for i in b:
            masked[i] = ""
        masked[b[0]] = str(len(b))  # single digit token == that many design positions
    spec = {
        "entities": [
            {"protein": {"id": "A", "sequence": "".join(masked)}}
        ]
    }
    import yaml
    with open(out, "w") as fh:
        yaml.safe_dump(spec, fh, default_flow_style=False, sort_keys=False)


def package(cand: dict, case_id: str, cdr: dict[str, list[int]], fr: list[int], seq: str) -> dict:
    d = OUT / "cases" / case_id
    (d / "intermediate_designs").mkdir(parents=True, exist_ok=True)
    write_cif(cand["file"], d / "backbone.cif", cand["chain"])
    shutil.copy(d / "backbone.cif", d / "intermediate_designs" / "design.cif")
    n = len(seq)
    np.savez(
        d / "intermediate_designs" / "design.npz",
        design_mask=np.eye(n, dtype=np.float32)[sum(cdr.values(), [])].sum(0),
        mol_type=np.zeros(n, dtype=np.int64),
        ss_type=np.zeros(n, dtype=np.int64),
        token_resolved_mask=np.ones(n, dtype=np.float32),
        binding_type=np.zeros(n, dtype=np.float32),
    )
    write_spec(seq, cdr, d / "design.yaml")
    with open(d / "provenance.json", "w") as fh:
        json.dump({"source_cif": cand["file"], "pdbid": cand["pdbid"],
                   "source_chain": cand["chain"], "imgt_scheme": True,
                   "cdr": cdr, "n_antigens": cand["n_antigens"]}, fh, indent=2)
    return d


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tmp").mkdir(exist_ok=True)

    print("[1/6] extracting VHH chains ...", flush=True)
    cands = extract()
    print(f"  {len(cands)} VHH-like chains", flush=True)
    fasta = OUT / "vhh_candidates.fasta"
    with open(fasta, "w") as fh:
        for c in cands:
            fh.write(f">{c['pdbid']}_{c['chain']}\n{c['seq']}\n")
    with open(OUT / "candidates.jsonl", "w") as fh:
        for c in cands:
            fh.write(json.dumps(c) + "\n")

    print("[2/6] clustering at 70% ...", flush=True)
    rep = cluster(fasta, OUT / "tmp")
    by_rep: dict[str, list[str]] = {}
    for m, r in rep.items():
        by_rep.setdefault(r, []).append(m)
    print(f"  {len(by_rep)} clusters", flush=True)

    print("[3/6] leakage check vs valid100 ...", flush=True)
    valid = valid100_seqs()
    leaked = leakage_check(valid, fasta, OUT / "tmp")
    print(f"  {len(leaked)} candidates excluded (>=85% id to valid100)", flush=True)

    print("[4/6] cluster-level split ...", flush=True)
    header2cand = {f"{c['pdbid']}_{c['chain']}": c for c in cands}
    reps = [r for r in by_rep if r not in leaked]
    reps = [r for r in reps if r in header2cand]
    # prefer clusters with antigen context; shuffle within strata
    with_ant = sorted([r for r in reps if header2cand[r]["n_antigens"] > 0])
    without = sorted([r for r in reps if header2cand[r]["n_antigens"] == 0])
    random.shuffle(with_ant)
    random.shuffle(without)
    picked = (with_ant + without)[: N_TRAIN + N_HELD]
    train_reps, held_reps = picked[:N_TRAIN], picked[N_TRAIN:]
    print(f"  {len(reps)} clean clusters; picked {len(train_reps)} train / {len(held_reps)} held-out", flush=True)

    print("[5/6] ANARCII numbering ...", flush=True)
    picks = [(r, "train") for r in train_reps] + [(r, "heldout") for r in held_reps]
    numbering = anarcii_number([(r, header2cand[r]["seq"]) for r, _ in picks])

    print("[6/6] packaging cases + manifest ...", flush=True)
    rows = []
    for i, (r, split) in enumerate(picks):
        c = header2cand[r]
        item = numbering.get(r)
        if item is None or "numbering" not in item:
            print(f"  SKIP {r}: no numbering")
            continue
        parsed = cdr_positions(item, c["seq"])
        if parsed is None:
            print(f"  SKIP {r}: numbering does not reconstruct sequence")
            continue
        cdr, fr = parsed
        lens = {k: len(v) for k, v in cdr.items()}
        if any(v < 3 for v in lens.values()) or len(fr) < 60:
            print(f"  SKIP {r}: bad CDRs {lens} fr={len(fr)}")
            continue
        case_id = f"sab2_{c['pdbid'].lower()}_{c['chain'].lower()}"
        seq = c["seq"]
        assert sorted(fr + cdr["cdr1"] + cdr["cdr2"] + cdr["cdr3"]) == list(range(len(seq))), case_id
        d = package(c, case_id, cdr, fr, seq)
        rows.append({
            "case_id": case_id,
            "structure_path": str(d / "backbone.cif"),
            "chain_id": "A",
            "full_sequence": seq,
            "design_positions": sorted(cdr["cdr1"] + cdr["cdr2"] + cdr["cdr3"]),
            "fr_positions": sorted(fr),
            "split": split,
            "seed_base": SEED + i,
            "design_mask_source": "SAbDab2 crystal + anarcii vnar IMGT CDR1/2/3 (round1_rl_split)",
        })

    with open(OUT / "rl_manifest_split.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    ntr = sum(1 for r in rows if r["split"] == "train")
    nhe = sum(1 for r in rows if r["split"] == "heldout")
    summary = dict(n_candidates=len(cands), n_clusters=len(by_rep),
                   n_leaked=len(leaked), n_train=ntr, n_heldout=nhe,
                   seed=SEED, rows=len(rows))
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
