#!/usr/bin/env bash
#SBATCH --job-name=vhh-xref
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/vhh-xref-%j.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/vhh-xref-%j.err

PY=/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python
cd /share/home/rongdingyi/programs/proteingen
$PY - <<'EOF'
import os, hashlib, gemmi, collections, json

# 1) extract all chain seqs from Sabdab2 cifs
d = '/share/data/limc/structure_lm_data/Sabdab2'
struct_seqs = {}   # seq -> [(pdbid, chainname, hexfile)]
nfiles = 0
for f in sorted(os.listdir(d)):
    if not f.endswith('.cif'):
        continue
    stem = f[4:-4]
    try:
        txt = bytes.fromhex(stem).decode()
        parts = txt.split('_')
        pdbid = parts[1] if len(parts) >= 4 else '?'
    except Exception:
        pdbid, parts = '?', []
    try:
        s = gemmi.read_structure(os.path.join(d, f))
        st = s[0]
    except Exception:
        continue
    nfiles += 1
    for ch in st:
        seq = gemmi.one_letter_code([r.name for r in ch])
        if not seq or len(seq) < 100:
            continue
        seq = seq.upper().replace('X', '')
        struct_seqs.setdefault(seq, []).append((pdbid, ch.name, f))
print(f'parsed {nfiles} cif files, {len(struct_seqs)} unique chain sequences')

seq_set = set(struct_seqs)
# chain length distribution of single-chain entries
vhh_like = {s: v for s, v in struct_seqs.items() if 110 <= len(s) <= 135}
print(f'VHH-like chains (110-135 aa): {len(vhh_like)} unique seqs')

# 2) scan plm_data vhh splits
def scan(path, name, limit=None):
    n = hit = 0
    hits = []
    with open(path) as fh:
        for line in fh:
            seq = line.strip()
            if not seq:
                continue
            n += 1
            if seq in seq_set:
                hit += 1
                if len(hits) < 5:
                    hits.append((seq[:40], struct_seqs[seq][:2]))
            if limit and n >= limit:
                break
    print(f'{name}: scanned {n} seqs, exact-match to Sabdab2 chains: {hit}')
    for h in hits:
        print('   ', h[0], h[1])
    return n, hit

scan('/share/data/plm_data/vhh.valid.txt', 'vhh.valid.txt')
scan('/share/data/plm_data/vhh.train.txt', 'vhh.train.txt', limit=500000)

json.dump({'unique_chain_seqs': len(struct_seqs),
           'vhh_like': len(vhh_like)},
          open('/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/vhh-xref-summary.json', 'w'))
EOF
echo DONE
