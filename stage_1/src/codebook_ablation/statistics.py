"""Paired capacity contrasts; identical seed/trip draws preserve the crossed design."""
import numpy as np
from hierarchical_rqvae.analyze import holm

def capacity_effects(baseline, candidate, seed_draws, trip_draws):
    b=np.asarray(baseline,dtype=float);h=np.asarray(candidate,dtype=float)
    if b.shape!=h.shape or b.ndim!=3 or not np.isfinite([b,h]).all():
        raise ValueError('expected finite aligned [capacity, seed, trip] arrays')
    if (b<0).any() or (h<0).any():raise ValueError('negative error')
    if seed_draws.ndim!=2 or trip_draws.ndim!=2 or len(seed_draws)!=len(trip_draws):
        raise ValueError('invalid paired draws')
    if (b.mean(-1)<=0).any():raise ValueError("zero baseline mean")
    estimates=1-h.mean(-1)/b.mean(-1)
    if not np.isfinite(estimates).all():raise ValueError('zero baseline mean')
    draws=[]
    for seeds,trips in zip(seed_draws,trip_draws):
        bm=b[:,seeds][:,:,trips].mean(-1);hm=h[:,seeds][:,:,trips].mean(-1)
        if (bm<=0).any():raise ValueError('zero resampled baseline')
        draws.append((1-hm/bm).mean(-1))
    return estimates,np.asarray(draws)

def summarize(baseline,candidate,replicates=10000,seed=271828,margin=.05):
    b=np.asarray(baseline,dtype=float);h=np.asarray(candidate,dtype=float)
    if b.shape!=h.shape or b.ndim!=3:raise ValueError('array shape mismatch')
    k,s,n=b.shape
    if k<2 or s<2 or n<2:raise ValueError('insufficient capacities, seeds, or trips')
    rng=np.random.default_rng(seed)
    sd=rng.integers(0,s,size=(replicates,s));td=rng.integers(0,n,size=(replicates,n))
    per_seed,draws=capacity_effects(b,h,sd,td)
    estimate=per_seed.mean(-1);interactions=estimate[1:]-estimate[0]
    interaction_draws=draws[:,1:]-draws[:,[0]]
    effects=np.concatenate([estimate,interactions])
    all_draws=np.concatenate([draws,interaction_draws],axis=1)
    # Descriptive centered bootstrap p values; final decisions use simultaneous intervals.
    p=(1+(np.abs(all_draws-effects)>=np.abs(effects)).sum(0))/(replicates+1)
    upper=np.quantile(draws,1-.05/k,axis=0)
    interaction_ci=np.quantile(interaction_draws,[.05/(2*(k-1)),1-.05/(2*(k-1))],axis=0).T
    return {'relative_improvement_per_seed':per_seed.tolist(),
        'mean_relative_improvement':estimate.tolist(),
        'seed_std':per_seed.std(-1,ddof=1).tolist(),
        'ci95':np.quantile(draws,[.025,.975],axis=0).T.tolist(),
        'simultaneous_one_sided_upper95':upper.tolist(),
        'interaction_vs_base':interactions.tolist(),
        'interaction_simultaneous_ci95':interaction_ci.tolist(),
        'centered_bootstrap_p':p.tolist(),'holm_p_all_capacity_and_interaction_contrasts':holm(p),
        'all_tested_capacity_benefit_upper_bounds_below_margin':bool((upper<margin).all()),
        'all_tested_interaction_intervals_inside_margin':bool((interaction_ci[:,0]>-margin).all() and (interaction_ci[:,1]<margin).all()),
        'replicates':replicates,'seed':seed,'margin':margin,
        'method':'crossed seed/trip bootstrap; identical seed draws and trajectory-ID draws across both methods and all capacities; equal-weight mean of per-seed relative improvements',
        'limitations':['Only three training seeds; bootstrap uncertainty is approximate.',
            'Confidence about tested nominal capacities does not establish adequate effective codebook utilization.',
            'Non-significance alone is not evidence of equivalence.']}
