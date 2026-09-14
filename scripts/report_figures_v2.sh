#!/usr/bin/env bash
#SBATCH --job-name=rl-report-figs
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/rl-report-figs-%j.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/rl-report-figs-%j.err
/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python -u /share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl/scripts/make_report_figures_v2.py
