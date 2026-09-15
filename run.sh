#!/usr/bin/env bash
# Gate-ordered entry point. Every hard gate exits non-zero on failure.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMD="${1:-help}"

run_gate() {
  echo "== [run.sh] $* =="
  "$@"
}

case "$CMD" in
  audit)
    run_gate bash "$ROOT/scripts/audit_repo.sh" "$ROOT/docs/AUDIT.md" ;;
  baseline)
    run_gate "$ROOT/scripts/submit_gate.sh" baseline ;;
  toy-reinforce)
    run_gate "$ROOT/scripts/submit_gate.sh" toy-reinforce ;;
  toy-grpo)
    run_gate "$ROOT/scripts/submit_gate.sh" toy-grpo ;;
  scorer-overfit)
    run_gate "$ROOT/scripts/submit_gate.sh" scorer-overfit ;;
  evaluate)
    run_gate "$ROOT/scripts/submit_gate.sh" evaluate ;;
  native-audit)
    run_gate "$ROOT/scripts/sbatch_native_smoke.sh" ;;
  native-base-pool)
    run_gate env SPLIT=train NUM=32 NSHARDS=4 sbatch --array=0-3 "$ROOT/scripts/sbatch_native_pool.sh" ;;
  native-build-pairs)
    run_gate /share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python "$ROOT/scripts/native_build_pairs.py" --split train ;;
  native-smoke)
    run_gate "$ROOT/scripts/sbatch_native_smoke.sh" ;;
  native-n1-rwr)
    run_gate "$ROOT/scripts/sbatch_native_train.sh" --method n1 --max-steps 500 ;;
  native-n2-dpo-full)
    run_gate "$ROOT/scripts/sbatch_native_train.sh" --method n2 --beta 10.0 --max-steps 500 ;;
  native-n3-dpo-fake)
    run_gate "$ROOT/scripts/sbatch_native_train.sh" --method n3 --beta 10.0 --max-steps 500 ;;
  native-n4-dpo-anchor)
    run_gate "$ROOT/scripts/sbatch_native_train.sh" --method n4 --beta 10.0 --max-steps 500 ;;
  native-eval)
    run_gate "$ROOT/scripts/sbatch_native_eval_v2.sh" ;;
  native-round1)
    # strict gate order (task book §100); every hard gate exits non-zero
    run_gate "$ROOT/scripts/sbatch_native_smoke.sh"
    run_gate env SPLIT=train NUM=32 NSHARDS=4 sbatch --array=0-3 "$ROOT/scripts/sbatch_native_pool.sh"
    run_gate /share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python "$ROOT/scripts/native_build_pairs.py" --split train
    run_gate "$ROOT/scripts/sbatch_native_train.sh" --method n1 --max-steps 500
    run_gate "$ROOT/scripts/sbatch_native_train.sh" --method n2 --beta 10.0 --max-steps 500
    run_gate "$ROOT/scripts/sbatch_native_train.sh" --method n3 --beta 10.0 --max-steps 500
    run_gate "$ROOT/scripts/sbatch_native_train.sh" --method n4 --beta 10.0 --max-steps 500
    ;;
  all-round1)
    run_gate bash "$ROOT/scripts/audit_repo.sh" "$ROOT/docs/AUDIT.md"
    run_gate "$ROOT/scripts/submit_gate.sh" baseline
    run_gate "$ROOT/scripts/submit_gate.sh" toy-reinforce
    run_gate "$ROOT/scripts/submit_gate.sh" toy-grpo
    run_gate "$ROOT/scripts/submit_gate.sh" scorer-overfit
    run_gate "$ROOT/scripts/submit_gate.sh" evaluate
    echo "round-1 gates finished; write docs/ROUND1_RESULTS.md and STOP" ;;
  next-repro)
    run_gate /usr/bin/python3 "$ROOT/scripts/next_reproduce_stats.py" ;;
  next-build-cf)
    run_gate /usr/bin/python3 "$ROOT/scripts/next_build_counterfactuals.py" ;;
  next-score-cf)
    run_gate "$ROOT/scripts/sbatch_next_cf_score.sh" ;;
  next-analyze-cf)
    run_gate "$ROOT/scripts/sbatch_next_cf_analyze.sh" ;;
  next-build-weights)
    run_gate /usr/bin/python3 "$ROOT/scripts/next_build_weights.py" ;;
  next-audit)
    run_gate "$ROOT/scripts/sbatch_next_b1_audit.sh" ;;
  next-analyze-audit)
    run_gate "$ROOT/scripts/sbatch_next_b1_analyze.sh" ;;
  next-smoke)
    run_gate "$ROOT/scripts/sbatch_next_smoke.sh" ;;
  next-b1-random-sparse)
    run_gate "$ROOT/scripts/sbatch_next_cf_dpo.sh" --variant random_sparse --max-steps 500 --checkpoint-every 50 --output-dir "$ROOT/runs/next_stage/full_random_sparse" --tag full-b1 ;;
  next-b2-shuffle)
    run_gate "$ROOT/scripts/sbatch_next_cf_dpo.sh" --variant shuffle --max-steps 500 --checkpoint-every 50 --output-dir "$ROOT/runs/next_stage/full_shuffle" --tag full-b2 ;;
  next-b3-cf)
    run_gate "$ROOT/scripts/sbatch_next_cf_dpo.sh" --variant cf --max-steps 500 --checkpoint-every 50 --output-dir "$ROOT/runs/next_stage/full_cf" --tag full-b3 ;;
  next-d1-hard)
    run_gate "$ROOT/scripts/sbatch_next_cf_dpo.sh" --variant uniform_all --temporal hard --sigma-seq 0.3137 --max-steps 500 --checkpoint-every 50 --output-dir "$ROOT/runs/next_stage/d_T1_hard" --tag d-t1 ;;
  next-d2-smooth)
    run_gate "$ROOT/scripts/sbatch_next_cf_dpo.sh" --variant uniform_all --temporal smooth --sigma-seq 0.3137 --max-steps 500 --checkpoint-every 50 --output-dir "$ROOT/runs/next_stage/d_T2_smooth" --tag d-t2 ;;
  next-select-ckpt)
    run_gate "$ROOT/scripts/sbatch_next_ckpt_eval.sh"
    run_gate /usr/bin/python3 "$ROOT/scripts/next_select_ckpt.py" ;;
  next-valid100)
    run_gate "$ROOT/scripts/sbatch_next_valid100_eval.sh" ;;
  next-v100-agg)
    run_gate /share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python "$ROOT/scripts/next_v100_agg.py" ;;
  next-refold-build)
    run_gate "$ROOT/scripts/sbatch_next_refold_build.sh" ;;
  next-refold-compare)
    run_gate /share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python "$ROOT/scripts/next_refold_compare.py" ;;
  paper-freeze-protocol)
    run_gate /share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python "$ROOT/experiments/paper_stage/scripts/verify_frozen_protocol.py" ;;
  paper-multiseed-n3)
    run_gate "$ROOT/scripts/sbatch_native_train.sh" --method n3 --beta 10.0 --max-steps 500 --seed 43 --output-dir "$ROOT/runs/paper_stage/n3_s43"
    run_gate "$ROOT/scripts/sbatch_native_train.sh" --method n3 --beta 10.0 --max-steps 500 --seed 44 --output-dir "$ROOT/runs/paper_stage/n3_s44" ;;
  paper-multiseed-shuffle)
    run_gate "$ROOT/scripts/sbatch_next_cf_dpo.sh" --variant shuffle --max-steps 500 --checkpoint-every 50 --seed 43 --output-dir "$ROOT/runs/paper_stage/shuffle_s43" --tag shuffle_s43
    run_gate "$ROOT/scripts/sbatch_next_cf_dpo.sh" --variant shuffle --max-steps 500 --checkpoint-every 50 --seed 44 --output-dir "$ROOT/runs/paper_stage/shuffle_s44" --tag shuffle_s44 ;;
  paper-multiseed-cf)
    run_gate "$ROOT/scripts/sbatch_next_cf_dpo.sh" --variant cf --max-steps 500 --checkpoint-every 50 --seed 43 --output-dir "$ROOT/runs/paper_stage/cf_s43" --tag cf_s43
    run_gate "$ROOT/scripts/sbatch_next_cf_dpo.sh" --variant cf --max-steps 500 --checkpoint-every 50 --seed 44 --output-dir "$ROOT/runs/paper_stage/cf_s44" --tag cf_s44 ;;
  paper-ablate-diffonly)
    run_gate "$ROOT/scripts/sbatch_next_cf_dpo.sh" --variant diff_only --weights "$ROOT/runs/paper_stage/weights/ablation_weights.json" --max-steps 500 --checkpoint-every 50 --output-dir "$ROOT/runs/paper_stage/diff_only" --tag diff_only ;;
  paper-ablate-drop)
    run_gate "$ROOT/scripts/sbatch_next_cf_dpo.sh" --variant drop_only --weights "$ROOT/runs/paper_stage/weights/ablation_weights.json" --max-steps 500 --checkpoint-every 50 --output-dir "$ROOT/runs/paper_stage/drop_only" --tag drop_only ;;
  paper-ablate-gain)
    run_gate "$ROOT/scripts/sbatch_next_cf_dpo.sh" --variant gain_only --weights "$ROOT/runs/paper_stage/weights/ablation_weights.json" --max-steps 500 --checkpoint-every 50 --output-dir "$ROOT/runs/paper_stage/gain_only" --tag gain_only ;;
  paper-query-budget)
    run_gate "$ROOT/scripts/sbatch_next_cf_dpo.sh" --variant 0.25 --weights "$ROOT/runs/paper_stage/weights/query_budget_weights.json" --max-steps 500 --checkpoint-every 50 --output-dir "$ROOT/runs/paper_stage/qb_025" --tag qb_025
    run_gate "$ROOT/scripts/sbatch_next_cf_dpo.sh" --variant 0.50 --weights "$ROOT/runs/paper_stage/weights/query_budget_weights.json" --max-steps 500 --checkpoint-every 50 --output-dir "$ROOT/runs/paper_stage/qb_050" --tag qb_050
    run_gate "$ROOT/scripts/sbatch_next_cf_dpo.sh" --variant 0.75 --weights "$ROOT/runs/paper_stage/weights/query_budget_weights.json" --max-steps 500 --checkpoint-every 50 --output-dir "$ROOT/runs/paper_stage/qb_075" --tag qb_075 ;;
  paper-structure-boltz2)
    run_gate bash -c "cd /share/home/rongdingyi/programs/proteingen/vhh_esmc_guidance && sbatch --array=0-3%4 scripts/cdr_all/sbatch_refold_main.sh '$ROOT/runs/paper_stage/structure_boltz2'" ;;
  paper-second-reward-audit)
    run_gate "$ROOT/experiments/paper_stage/scripts/sbatch_second_reward_audit.sh" ;;
  paper-second-reward-pilot)
    run_gate /share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python "$ROOT/experiments/paper_stage/scripts/run_second_reward.py" all ;;
  paper-structure-protenix)
    run_gate bash -c "cd /share/home/rongdingyi/programs/proteingen/vhh_esmc_guidance && bash scripts/submit_protenix_v2_array.sh '$ROOT/runs/paper_stage/structure_protenix/manifest.csv' valid100" ;;
  paper-report)
    run_gate /share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python "$ROOT/experiments/paper_stage/scripts/make_paper_tables.py"
    run_gate /share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python "$ROOT/experiments/paper_stage/scripts/make_paper_figures.py" ;;
  cf-opsd-audit)
    run_gate /usr/bin/python3 "$ROOT/scripts/cf_opsd_audit.py" ;;
  cf-opsd-rollout)
    run_gate "$ROOT/scripts/sbatch_cf_opsd_rollout.sh" ;;
  cf-opsd-query-probe)
    run_gate /usr/bin/python3 "$ROOT/scripts/cf_opsd_probe_queries.py" ;;
  cf-opsd-target-probe)
    run_gate "$ROOT/scripts/sbatch_cf_opsd_target.sh" ;;
  cf-opsd-target-report)
    run_gate /usr/bin/python3 "$ROOT/scripts/cf_opsd_target_report.py" ;;
  cf-opsd-realization-probe)
    run_gate "$ROOT/scripts/sbatch_cf_opsd_realization.sh" ;;
  cf-opsd-static)
    run_gate "$ROOT/scripts/sbatch_cf_opsd_static.sh" ;;
  cf-dpo-mini-matched)
    run_gate "$ROOT/scripts/sbatch_cf_opsd_cfdpo_mini.sh" ;;
  cf-opsd-static-eval)
    run_gate "$ROOT/scripts/sbatch_cf_opsd_static_eval.sh" ;;
  cf-opsd-onpolicy)
    run_gate "$ROOT/scripts/sbatch_cf_opsd_onpolicy.sh" ;;
  cf-opsd-compare)
    run_gate /share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python "$ROOT/scripts/cf_opsd_static_eval.py" ;;
  cf-opsd-report)
    run_gate /usr/bin/python3 "$ROOT/scripts/cf_opsd_report.py" ;;
  cfd2-build-graph)
    run_gate "$ROOT/experiments/cf_dpo_v2/scripts/sbatch_build_graph.sh" ;;
  cfd2-exp2)
    run_gate "$ROOT/experiments/cf_dpo_v2/scripts/sbatch_exp2_small.sh" ;;
  cfd2-train-signed)
    run_gate env VARIANT=signed sbatch "$ROOT/experiments/cf_dpo_v2/scripts/sbatch_signed_pilot.sh" ;;
  cfd2-train-v2)
    run_gate env VARIANT=v2 sbatch "$ROOT/experiments/cf_dpo_v2/scripts/sbatch_signed_pilot.sh" ;;
  cfd2-proxy)
    run_gate "$ROOT/experiments/cf_dpo_v2/scripts/sbatch_proxy_validation.sh" ;;
  cfd2-report)
    run_gate "$ROOT/experiments/cf_dpo_v2/scripts/sbatch_exp4_report.sh" ;;
  slcf-build-edges)
    run_gate env STAGE=build-edges sbatch "$ROOT/experiments/signed_local/scripts/sbatch_cpu.sh" ;;
  slcf-audit-edges)
    run_gate env STAGE=audit-edges sbatch "$ROOT/experiments/signed_local/scripts/sbatch_cpu.sh" ;;
  slcf-protocol)
    run_gate env STAGE=protocol sbatch "$ROOT/experiments/signed_local/scripts/sbatch_cpu.sh" ;;
  slcf-report)
    run_gate env STAGE=report sbatch "$ROOT/experiments/signed_local/scripts/sbatch_cpu.sh" ;;
  slcf-manifold)
    run_gate env STAGE=manifold sbatch "$ROOT/experiments/signed_local/scripts/sbatch_gpu.sh" ;;
  slcf-one-edge)
    run_gate env STAGE=one-edge sbatch "$ROOT/experiments/signed_local/scripts/sbatch_gpu.sh" ;;
  slcf-diagnostics)
    run_gate env STAGE=diagnostics sbatch "$ROOT/experiments/signed_local/scripts/sbatch_gpu.sh" ;;
  slcf-smoke)
    run_gate env STAGE=smoke sbatch "$ROOT/experiments/signed_local/scripts/sbatch_gpu.sh" ;;
  slcf-pilot)
    run_gate env STAGE=pilot ARM=${2:-main} sbatch "$ROOT/experiments/signed_local/scripts/sbatch_gpu.sh" ;;
  slcf-pilot-all)
    for arm in cf local neg main shuffle; do
      run_gate env STAGE=pilot ARM=$arm sbatch "$ROOT/experiments/signed_local/scripts/sbatch_gpu.sh"
    done ;;
  slcf-pilot-eval)
    run_gate env STAGE=pilot-eval sbatch "$ROOT/experiments/signed_local/scripts/sbatch_gpu.sh" ;;
  slcf-heldout)
    run_gate env STAGE=heldout sbatch "$ROOT/experiments/signed_local/scripts/sbatch_gpu.sh" ;;
  slcf-full)
    run_gate env STAGE=full ARM=${2:-f1} sbatch "$ROOT/experiments/signed_local/scripts/sbatch_gpu.sh" ;;
  slcf-full-all)
    for arm in f0 f1 f2; do
      run_gate env STAGE=full ARM=$arm sbatch "$ROOT/experiments/signed_local/scripts/sbatch_gpu.sh"
    done ;;
  *)
    echo "usage: bash run.sh {audit|baseline|toy-reinforce|toy-grpo|scorer-overfit|evaluate|all-round1|next-*|cf-opsd-*|cfd2-*|slcf-*}" >&2
    exit 2 ;;
esac
