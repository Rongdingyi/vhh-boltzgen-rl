#!/usr/bin/env bash
#SBATCH --job-name=refold5-analyze
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/refold5-analyze-%j.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/refold5-analyze-%j.err
/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python -u /share/home/rongdingyi/programs/proteingen/vhh_esmc_guidance/scripts/cdr_all/analyze_refold.py --design-dir /share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl/runs/round1_rl_split/refold5_design --manifest /share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl/runs/round1_rl_split/refold5_design/manifest.csv --out-prefix /share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl/runs/round1_rl_split/refold5_results --group-by label
