#!/usr/bin/env python3
"""Audit existing CF-DPO attribution CSVs and test exact finite-state identities.
No protein model is trained or run by this script.
Usage:
  python audit_and_math_checks.py --credit-csv residue_credit.csv \
    --pair-csv pair_credit_summary.csv --outdir results
Dependencies: numpy, pandas.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def audit(credit_path: Path, pair_path: Path, out: Path) -> dict:
    df = pd.read_csv(credit_path)
    pairs = pd.read_csv(pair_path)
    required = {'pair_id', 'case_id', 'c_drop', 'c_gain', 'c_avg', 'c_cons'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing credit columns: {required - set(df.columns)}')
    if not {'pair_id', 'reward_gap'}.issubset(pairs.columns):
        raise ValueError('Pair CSV needs pair_id and reward_gap')
    if pairs.pair_id.duplicated().any():
        raise ValueError('Pair IDs must be unique in the pair summary')
    if not np.isfinite(df[['c_drop','c_gain','c_avg','c_cons']].to_numpy()).all():
        raise ValueError('Non-finite credit values')
    if not set(df.pair_id).issubset(set(pairs.pair_id)):
        raise ValueError('Unmatched pair IDs')
    n = len(df)
    sign_rows = []
    for tol in (0., 1e-6, .05, .1):
        pos = (df.c_drop > tol) & (df.c_gain > tol)
        neg = (df.c_drop < -tol) & (df.c_gain < -tol)
        flip = ((df.c_drop > tol) & (df.c_gain < -tol)) | ((df.c_drop < -tol) & (df.c_gain > tol))
        for label, mask in [('both_positive',pos),('both_negative',neg),('context_sign_flip',flip),('near_zero_or_small',~(pos|neg|flip))]:
            sign_rows.append(dict(tolerance=tol, category=label, count=int(mask.sum()), fraction=float(mask.mean()), n_pairs=int(df.loc[mask,'pair_id'].nunique())))
    pd.DataFrame(sign_rows).to_csv(out/'signed_credit_counts.csv', index=False)
    row_stats = []
    for pid, g in df.groupby('pair_id',sort=True):
        rec = dict(pair_id=pid, case_id=str(g.case_id.iloc[0]), n_diff=len(g))
        k = int(math.ceil(.3 * len(g)))
        for name, values in [('positive_conservative',g.c_cons.to_numpy()),('absolute_average',np.abs(g.c_avg.to_numpy())),('absolute_drop',np.abs(g.c_drop.to_numpy())),('absolute_gain',np.abs(g.c_gain.to_numpy()))]:
            total = float(values.sum())
            rec['top30_mass_'+name] = float(np.sort(values)[-k:].sum()/total) if total > 0 else np.nan
        rec['best_reversion_gain'] = max(0., float(-g.c_drop.min()))
        rec['has_negative_both_tol005'] = bool(((g.c_drop < -.05)&(g.c_gain < -.05)).any())
        rec['has_sign_flip_tol005'] = bool((((g.c_drop > .05)&(g.c_gain < -.05))|((g.c_drop < -.05)&(g.c_gain > .05))).any())
        row_stats.append(rec)
    ps = pd.DataFrame(row_stats)
    ps.to_csv(out/'per_pair_reaudit.csv',index=False)
    reversion = []
    for tol in (0.,.05,.1):
        mask = df.c_drop < -tol
        reversion.append(dict(tolerance=tol, improving_queries=int(mask.sum()), pairs_with_better_than_winner=int(df.loc[mask,'pair_id'].nunique())))
    merged = df.merge(pairs[['pair_id','reward_gap']],on='pair_id',validate='many_to_one')
    gain_beats = merged.c_gain > merged.reward_gap + .05
    best = ps.best_reversion_gain
    result = {
        'inputs': {'credit_csv':str(credit_path),'credit_sha256':sha256(credit_path),'pair_csv':str(pair_path),'pair_sha256':sha256(pair_path)},
        'n_pairs':int(df.pair_id.nunique()),'n_cases':int(df.case_id.nunique()),'n_differing_residue_events':n,
        'sign_counts':sign_rows,
        'better_than_winner_reversions':reversion,
        'loser_mutation_beats_winner_tol005':{'queries':int(gain_beats.sum()),'pairs':int(merged.loc[gain_beats,'pair_id'].nunique())},
        'best_reversion_gain_per_pair':{'mean':float(best.mean()),'median':float(best.median()),'q25':float(best.quantile(.25)),'q75':float(best.quantile(.75)),'max':float(best.max())},
        'median_top30_mass':{col.replace('top30_mass_',''):float(ps[col].median()) for col in ps.columns if col.startswith('top30_mass_')},
        'interpretation_limits':['All effects are of the fixed classifier on selected training pairs, not biological causal effects.','0.05 and 0.1 are diagnostic tolerances, not biological thresholds.','94% is mass of clipped conservative-positive credit, not explained total reward gap.','These are audits of existing CSV data, not additional protein-model experiments.']
    }
    (out/'signed_credit_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def logsumexp(a: np.ndarray) -> float:
    m = float(np.max(a))
    return m + float(np.log(np.exp(a-m).sum()))


def math_checks() -> dict:
    # Counterexample: exact sum attribution, but sign flips in each background.
    def F(z):
        x,y=z
        return 2*x+2*y-3*x*y
    rewards = {''.join(map(str,z)):F(z) for z in itertools.product((0,1),repeat=2)}
    drop = np.array([F((1,1))-F((0,1)), F((1,1))-F((1,0))],dtype=float)
    gain = np.array([F((1,0))-F((0,0)), F((0,1))-F((0,0))],dtype=float)
    avg = .5*(drop+gain)
    cons = np.where((drop>0)&(gain>0),np.minimum(drop,gain),0.)
    assert np.allclose(avg.sum(),F((1,1))-F((0,0)))
    assert np.all(cons == 0) and np.all(drop*gain < 0)
    # Third-order interactions can cancel in the endpoint residual.
    def F3(z):
        a,b,c,d=z
        return 10*a+a*b*c-a*b*d
    zeros=np.zeros(4,dtype=int); ones=np.ones(4,dtype=int)
    av3=[]
    for i in range(4):
        ei=zeros.copy();ei[i]=1
        oi=ones.copy();oi[i]=0
        av3.append(.5*(F3(ones)-F3(oi)+F3(ei)-F3(zeros)))
    residual3=float(F3(ones)-F3(zeros)-sum(av3))
    assert abs(residual3) < 1e-12
    # Exact finite model: S in 8 states, G in 4 geometric representatives.
    rng=np.random.default_rng(130914)
    p0=rng.uniform(.2,1.,(8,4));p0/=p0.sum()
    seq=np.array(list(itertools.product((0,1),repeat=3)))
    R=1.2*seq[:,0]+.8*seq[:,1]-.3*seq[:,2]-1.4*seq[:,0]*seq[:,1]+.7*np.prod(seq,axis=1)
    beta=1.7
    pstar=p0*np.exp(R[:,None]/beta);pstar/=pstar.sum()
    q=rng.uniform(.1,1.5,(8,4));q/=q.sum()
    p0s=p0.sum(1);qs=q.sum(1);pstars=pstar.sum(1)
    p0c=p0/p0s[:,None];qc=q/qs[:,None]
    def KL(a,b): return float(np.sum(a*(np.log(a)-np.log(b))))
    def J(a): return float(np.sum(a*R[:,None])-beta*KL(a,p0))
    joint=KL(q,pstar)
    seq_kl=KL(qs,pstars)
    cond_kl=sum(qs[s]*KL(qc[s],p0c[s]) for s in range(8))
    assert abs(joint-seq_kl-cond_kl)<1e-12
    assert abs(J(pstar)-J(q)-beta*joint)<1e-12
    assert np.max(np.abs(pstar/pstars[:,None]-p0c))<1e-12
    # Marginal density ratio and conditional KL identities.
    lr=np.log(q/p0)
    errors=[]; reverse_errors=[]; forward_errors=[]
    for s in range(8):
        v=logsumexp(np.log(p0c[s])+lr[s])
        errors.append(abs(v-np.log(qs[s]/p0s[s])))
        reverse_errors.append(abs((v-np.sum(p0c[s]*lr[s]))-KL(p0c[s],qc[s])))
        forward_errors.append(abs((np.sum(qc[s]*lr[s])-v)-KL(qc[s],p0c[s])))
    assert max(errors+reverse_errors+forward_errors)<1e-12
    # Connected intervention/tie edges recover the reward potential up to a constant.
    n=32; edges=[]
    for s in range(8):
        for g in range(1,4): edges.append((4*s,4*s+g))
        for t in range(s+1,8):
            if np.sum(seq[s]!=seq[t])==1: edges.append((4*s,4*t))
    incidence=np.zeros((len(edges),n));y=np.zeros(len(edges))
    for k,(a,b) in enumerate(edges):
        incidence[k,b]=1;incidence[k,a]=-1;y[k]=R[b//4]-R[a//4]
    h,*_=np.linalg.lstsq(np.vstack([incidence,np.ones((1,n))]),np.r_[y,0.],rcond=None)
    graph_fit_error=float(np.max(np.abs(incidence@h-y)))
    pred=p0.flatten()*np.exp(h/beta);pred/=pred.sum()
    assert graph_fit_error<1e-12
    assert np.max(np.abs(pred-pstar.flatten()))<1e-12
    return {
        'status':'passed','type':'exact finite-state arithmetic checks, not protein experiments',
        'sign_reversal_example':{'reward_table':rewards,'drop':drop.tolist(),'gain':gain.tolist(),'average_credit':avg.tolist(),'conservative_credit':cons.tolist(),'endpoint_residual':float(F((1,1))-F((0,0))-avg.sum())},
        'high_order_cancellation':{'function':'10 z1 + z1 z2 z3 - z1 z2 z4','endpoint_residual':residual3,'has_nonzero_third_order_terms':True},
        'ideal_distribution_checks':{'objective_gap_error':abs(J(pstar)-J(q)-beta*joint),'kl_chain_error':abs(joint-seq_kl-cond_kl),'marginal_ratio_max_error':max(errors),'reverse_conditional_kl_max_error':max(reverse_errors),'forward_conditional_kl_max_error':max(forward_errors),'graph_potential_fit_error':graph_fit_error,'graph_recovered_distribution_max_error':float(np.max(np.abs(pred-pstar.flatten())))},
        'limits':['Density ratios in these checks are exact. Native BoltzGen denoising errors are not exact log densities.','Finite-state enumeration does not demonstrate native conditional sampling feasibility or improved protein generation.']
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--credit-csv',type=Path,required=True)
    ap.add_argument('--pair-csv',type=Path,required=True)
    ap.add_argument('--outdir',type=Path,required=True)
    args=ap.parse_args();args.outdir.mkdir(parents=True,exist_ok=True)
    res=audit(args.credit_csv,args.pair_csv,args.outdir)
    checks=math_checks()
    (args.outdir/'math_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'cases':res['n_cases'],'pairs':res['n_pairs'],'events':res['n_differing_residue_events'],'median_best_reversion_gain':res['best_reversion_gain_per_pair']['median'],'math_checks':checks['status']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
