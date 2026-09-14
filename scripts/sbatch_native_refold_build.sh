#!/usr/bin/env bash
#SBATCH --job-name=native-refold-build
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/native-refold-build-%j.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/native-refold-build-%j.err
/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python -u /share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl/scripts/native_refold_build.py
