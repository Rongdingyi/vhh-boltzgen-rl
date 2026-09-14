#!/usr/bin/env bash
#SBATCH --job-name=refold5-build
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/refold5-build-%j.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/refold5-build-%j.err
/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python -u /share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl/scripts/build_refold5_inputs.py
