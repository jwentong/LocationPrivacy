"""Reproducible three-group experiment suite, conditional on a frozen CSI bank.

Measured model: fixed cell centers, uniform prior, IID within-cell CSI bootstrap.
Array experiment: separately labelled ideal geometric Rician simulations.
"""
import hashlib
import json
import time
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import beta, ncx2, t

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/basic_v1'
FIG = OUT/'figures'
CONFIG = dict(version='basic_v1', seed=20261029, alpha=.1, r0_m=10,
              eta_grid=np.geomspace(.1, 10, 21).tolist(), delta_selection=.025,
              delta_confirmation=.025, Ns=20000, Nc=100000,
              repeats=500, reliability_N=[200, 500, 2000, 5000, 20000],
              D_values=[.03, .05, .1, .2], eta_caps=[.3, 1., 3., 10.],
              mechanism_N=100000, source_slots=5000000, source_batch=10000,
              reference_snr_db=10, spectral_efficiency=1,
              geometric_N=12000, geometric_kappa=10, geometric_alpha=.1,
              geometry_reference_distance=150, geometric_pathloss_exponent=2,
              geometry_single_element_snr_db=-5,
              geometry_equal_aligned_snr_db=13.061799739838872,
              geometry_arrays=[1,4,8,16])


def F(q, a=.1):
    q = np.asarray(q)
    return a*q/((1-(1-a)*q)*(1-(1-2*a)*q))


def intervals(qhat, n, pairs, delta=.025):
    d = np.sqrt(np.log(2*pairs/delta)/(2*n))
    return F(np.maximum(0,qhat-d)).mean(axis=-1), F(np.minimum(1,qhat+d)).mean(axis=-1)


def projection_cdf(x, epsilon):
    """Projection of 3D isotropic Laplace on any unit direction."""
    y=epsilon*np.abs(x)
    tail=(np.minimum(y,1e300)+2)*np.exp(-y)/4
    return np.where(x>=0,1-tail,tail)


def exact_confusion(candidate, epsilon):
    u=(candidate[-1]-candidate[0]).astype(float)
    u/=np.linalg.norm(u)
    coordinate=(candidate-candidate[0])@u
    residual=candidate-candidate[0]-coordinate[:,None]*u
    if np.max(np.abs(residual))>1e-9 or not np.all(np.diff(coordinate)>0):
        raise ValueError('Exact MAP integration requires ordered collinear sites')
    borders=np.r_[-np.inf,(coordinate[:-1]+coordinate[1:])/2,np.inf]
    answer=[]
    for x in coordinate:
        cdf=projection_cdf(borders-x,epsilon)
        answer.append(np.diff(cdf))
    return np.array(answer)


def model():
    with np.load(ROOT/'data/frozen/map.npz') as d:
        d={k:d[k] for k in d.files}
    with np.load(ROOT/'data/aligned/session.npz') as full:
        calibration=full['h'][d['train_id']]
    # Calibrate ONLY on train CSI, never on test outcomes or a budget curve.
    power=float(np.median(np.sum(np.abs(calibration)**2,axis=1)))
    scale=10**(CONFIG['reference_snr_db']/10)/power
    candidate=np.unique(np.floor(d['test_xyz']/10).astype(int),axis=0)*10+5
    group=cKDTree(candidate).query(d['test_xyz'])[1]
    beam_id=cKDTree(d['map_xyz']).query(candidate)[1]
    w=d['beams'][beam_id]
    gain=np.abs(d['test_h']@w.conj().T)**2
    outage=scale*gain < 2**CONFIG['spectral_efficiency']-1
    conditional=np.array([outage[group==j].mean(axis=0) for j in range(len(candidate))])
    etas=np.array(CONFIG['eta_grid'])
    confusion=np.array([exact_confusion(candidate,e/CONFIG['r0_m']) for e in etas])
    q=np.einsum('ejk,jk->ej',confusion,conditional)
    # Raw-report beam baseline queries the same frozen training map directly.
    np.savez_compressed(OUT/'empirical_model.npz',candidate=candidate,conditional_outage=conditional,
                        confusion=confusion,q=q,eta=etas,weights=np.ones(len(candidate))/len(candidate),
                        scenario_counts=np.bincount(group),map_xyz=d['map_xyz'],beams=d['beams'])
    CONFIG['calibration_train_power']=power
    CONFIG['snr_scale']=scale
    CONFIG['scenario_count']=len(candidate)
    CONFIG['scenario_counts']=np.bincount(group).tolist()
    CONFIG['candidate_xyz']=candidate.tolist()
    CONFIG['map_sha256']=hashlib.sha256((ROOT/'data/frozen/map.npz').read_bytes()).hexdigest()
    return d,candidate,group,w,conditional,confusion,q,scale


def paired_mechanism(d,candidate,group,w,conditional,confusion,q,scale):
    rng=np.random.default_rng(CONFIG['seed'])
    n=CONFIG['mechanism_N']; etas=np.array(CONFIG['eta_grid'])
    draws=[]
    for j in range(len(candidate)):
        z=rng.normal(size=(n,3)); z/=np.linalg.norm(z,axis=1,keepdims=True)
        draws.append((rng.gamma(3,1,n)[:,None]*z,
                      rng.choice(np.flatnonzero(group==j),n)))
    rows=[]
    for e,eta in enumerate(etas):
        for j in range(len(candidate)):
            noise,ids=draws[j]
            report=candidate[j]+noise/(eta/CONFIG['r0_m'])
            inferred=cdist(report,candidate).argmin(axis=1)
            hh=d['test_h'][ids]
            ww=w[inferred]
            gain=np.abs(np.sum(ww.conj()*hh,axis=1))**2
            qhat=float(np.mean(scale*gain<1))
            direct_id=cKDTree(d['map_xyz']).query(report)[1]
            direct_gain=np.abs(np.sum(d['beams'][direct_id].conj()*hh,axis=1))**2
            distance=np.linalg.norm(candidate[inferred]-candidate[j],axis=1)
            rows.append(dict(eta=eta,scenario=j,q_exact=q[e,j],q_mc=qhat,q_selection=float(np.mean(scale*gain[:CONFIG['Ns']]<1)),aoii_exact=F(q[e,j]),
                             q_direct=float(np.mean(scale*direct_gain<1)),
                             mean_error_m=float(distance.mean()),
                             mean_normalized_gain=float(np.mean(gain/np.sum(np.abs(hh)**2,axis=1)))))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'mechanism_per_scenario.csv',index=False)
    agg=frame.groupby('eta',as_index=False).agg(q_exact=('q_exact','mean'),q_mc=('q_mc','mean'),
         aoii=('aoii_exact','mean'),q_direct=('q_direct','mean'),error_m=('mean_error_m','mean'),
         gain=('mean_normalized_gain','mean'))
    agg['aoii_direct']=frame.assign(a=F(frame.q_direct)).groupby('eta').a.mean().to_numpy()
    agg.to_csv(OUT/'mechanism.csv',index=False)
    return agg


def simulate_source(a,q,seed,slots=1000000,batch=10000):
    rng=np.random.default_rng(seed)
    # Direct state and estimate recursion, independent of analytical age chain.
    flip=rng.random(slots+batch)<a
    success=rng.random(slots+batch)>=q
    source=estimate=age=0; total=0; means=[]
    for i in range(slots+batch):
        source ^= int(flip[i])
        if success[i]: estimate=source
        age=0 if source==estimate else age+1
        total+=age
        if (i+1)%batch==0:
            if i+1>batch:means.append(total/batch)
            total=0
    return float(np.mean(means)),float(np.std(means,ddof=1)/np.sqrt(len(means)))


def source_validation(q):
    rows=[]
    for a in [.05,.1,.3,.5]:
        for outage in [.1,.3,.5,.7,.9]:
            mean,se=simulate_source(a,outage,CONFIG['seed']+len(rows),CONFIG['source_slots'],CONFIG['source_batch'])
            rows.append(dict(alpha=a,q=outage,theory=float(F(outage,a)),simulation=mean,se=se,
                             halfwidth95=float(t.ppf(.975,CONFIG['source_slots']//CONFIG['source_batch']-1)*se),kind='controlled'))
    # End-to-end values: actual report inference and empirical link margins
    # define exact Bernoulli q; independent slots have that same distribution.
    for e in [0,10,20]:
        for j in range(q.shape[1]):
            mean,se=simulate_source(.1,float(q[e,j]),CONFIG['seed']+len(rows),CONFIG['source_slots'],CONFIG['source_batch'])
            rows.append(dict(alpha=.1,q=float(q[e,j]),theory=float(F(q[e,j])),simulation=mean,se=se,
                             halfwidth95=float(t.ppf(.975,CONFIG['source_slots']//CONFIG['source_batch']-1)*se),kind='measured_marginal',eta=CONFIG['eta_grid'][e],scenario=j))
    frame=pd.DataFrame(rows);frame['relative_error']=abs(frame.simulation-frame.theory)/frame.theory
    frame.to_csv(OUT/'theory_validation.csv',index=False)
    return frame


def steering(x,n):
    u=x/np.linalg.norm(x,axis=1,keepdims=True)
    rx,ry=np.meshgrid(np.arange(n),np.arange(n),indexing='ij')
    r=np.c_[rx.ravel(),ry.ravel()]
    return np.exp(1j*np.pi*(u[:,:2]@r.T))


def array_experiment():
    candidate=np.array([[x,y,z] for x in [-100,-50,0,50,100]
                         for y in [-100,-50,0,50,100] for z in [80,120,160]],float)
    jcount=len(candidate); etas=np.array(CONFIG['eta_grid']);n=CONFIG['geometric_N']
    rng=np.random.default_rng(CONFIG['seed']+100)
    # Report-confusion probabilities estimated once and reused for all arrays.
    confusion=np.zeros((len(etas),jcount,jcount))
    for j,x in enumerate(candidate):
        unit=rng.normal(size=(n,3));unit/=np.linalg.norm(unit,axis=1,keepdims=True)
        noise=unit*rng.gamma(3,1,n)[:,None]
        for e,eta in enumerate(etas):
            nearest=cdist(x+noise/(eta/10),candidate).argmin(axis=1)
            confusion[e,j]=np.bincount(nearest,minlength=jcount)/n
    rows=[]; raw={}
    path=(np.linalg.norm(candidate,axis=1)/150)**-2
    for side in CONFIG['geometry_arrays']:
        m=side**2; a=steering(candidate,side);cor=np.abs(a@a.conj().T/np.sqrt(m))**2
        kappa=CONFIG['geometric_kappa']
        for mode in ['fixed_power','equal_aligned_snr']:
            single_snr=(10**(CONFIG['geometry_single_element_snr_db']/10) if mode=='fixed_power'
                        else 10**(CONFIG['geometry_equal_aligned_snr_db']/10)/m)
            variance=single_snr*path[:,None]/(kappa+1)
            # Exact Rician projection CDF conditional on inferred location.
            qjk=ncx2.cdf(2/variance,df=2,nc=2*kappa*cor)
            conditional_q=np.einsum('ejk,jk->ej',confusion,qjk)
            oracle=F(np.diag(qjk)).mean()
            for e,eta in enumerate(etas):
                rows.append(dict(M=m,side=side,mode=mode,eta=eta,q=conditional_q[e].mean(),
                                 aoii=F(conditional_q[e]).mean(),oracle_aoii=oracle))
            raw[f'{side}_{mode}']=conditional_q
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'array.csv',index=False)
    # kappa=0: analytical beam-independent Rayleigh control.
    rayleigh=1-np.exp(-1/(10**(-5/10)*path))
    pd.DataFrame(dict(eta=etas,q=np.full(len(etas),rayleigh.mean()),
                      aoii=np.full(len(etas),F(rayleigh).mean()))).to_csv(OUT/'rayleigh_control.csv',index=False)
    np.savez_compressed(OUT/'array_raw.npz',candidate=candidate,confusion=confusion,**raw)
    return frame


def selection(q):
    rng=np.random.default_rng(CONFIG['seed']+200); etas=np.array(CONFIG['eta_grid'])
    jcount=q.shape[1]; pairs=q.size; n=CONFIG['Ns'];nc=CONFIG['Nc']
    paired=pd.read_csv(OUT/'mechanism_per_scenario.csv')
    qhat=paired.pivot(index='eta',columns='scenario',values='q_selection').to_numpy()
    if qhat.shape!=q.shape: raise ValueError('Selection/outage grid mismatch')
    lo,hi=intervals(qhat,n,pairs)
    exact=F(q).mean(axis=1);point=F(qhat).mean(axis=1)
    curve=pd.DataFrame(dict(eta=etas,J_exact=exact,J_estimate=point,L=lo,U=hi))
    curve.to_csv(OUT/'selection_curve.csv',index=False)
    rows=[]
    for cap in CONFIG['eta_caps']:
        feasible=np.flatnonzero(etas<=cap+1e-12);idx=feasible[np.argmin(hi[feasible])]
        reference=feasible[np.argmin(exact[feasible])]
        qconfirm=rng.binomial(nc,q[idx])/nc
        lc,uc=intervals(qconfirm,nc,jcount)
        rows.append(dict(problem='P1',threshold=cap,eta=etas[idx],J_exact=exact[idx],L_confirm=float(lc),
                         U_confirm=float(uc),eta_reference=etas[reference],regret=exact[idx]-exact[reference],
                         G=hi[idx]-lo[feasible].min(),status='privacy_feasible'))
    for target in CONFIG['D_values']:
        feasible=np.flatnonzero(hi<=target)
        if not len(feasible):
            rows.append(dict(problem='P2',threshold=target,status='infeasible' if lo.min()>target else 'unconfirmed'))
            continue
        idx=feasible[0]; qconfirm=rng.binomial(nc,q[idx])/nc
        lc,uc=intervals(qconfirm,nc,jcount)
        reference=np.flatnonzero(exact<=target)
        rows.append(dict(problem='P2',threshold=target,eta=etas[idx],J_exact=exact[idx],L_confirm=float(lc),
                         U_confirm=float(uc),eta_reference=etas[reference[0]] if len(reference) else np.nan,
                         status='confirmed' if uc<=target else 'unconfirmed'))
    pd.DataFrame(rows).to_csv(OUT/'budget_selection.csv',index=False)
    # Confirmation binomial counts are exact marginal simulations of IID empirical links.
    # The exact marginal Bernoulli model isolates interval/selection reliability.
    # Independent budget streams are valid but omit paired-simulation efficiency.
    rep=CONFIG['repeats'];target=.05; records=[]; macro=[]
    for n in CONFIG['reliability_N']:
        samples=rng.binomial(n,q,size=(rep,*q.shape))/n
        lower,upper=intervals(samples,n,pairs)
        estimate=F(samples).mean(axis=-1)
        for r in range(rep):
            coverage=bool(np.all((lower[r]<=exact)&(exact<=upper[r])))
            for method,cost in [('LAPIS',upper[r]),('point_estimate',estimate[r])]:
                feasible=np.flatnonzero(cost<=target)
                chosen=int(feasible[0]) if len(feasible) else -1
                violation=chosen>=0 and exact[chosen]>target
                confirmed=False
                if chosen>=0:
                    qc=rng.binomial(nc,q[chosen])/nc
                    lc,uc=intervals(qc,nc,jcount)
                    confirmed=bool(uc<=target)
                macro.append(dict(Ns=n,repeat=r,method=method,selected=chosen,eta=etas[chosen] if chosen>=0 else np.nan,
                                  J_exact=exact[chosen] if chosen>=0 else np.nan,violation=violation,
                                  issued=chosen>=0,confirmation_passed=confirmed,simultaneous_coverage=coverage,
                                  mean_width=float(np.mean(upper[r]-lower[r]))))
        block=pd.DataFrame([x for x in macro if x['Ns']==n])
        for method,b in block.groupby('method'):
            count=int(b.violation.sum());issued=int(b.issued.sum())
            records.append(dict(Ns=n,method=method,repeats=rep,issued=issued,violations=count,
                                violation_rate=count/rep,conditional_violation_rate=count/issued if issued else np.nan,
                                confirmation_rate=float(b.confirmation_passed.mean()),
                                coverage=float(b.simultaneous_coverage.mean()),
                                false_issue_upper95=float(beta.ppf(.975,count+1,rep-count)) if count<rep else 1.,
                                mean_width=float(b.mean_width.mean()),mean_eta=float(b.eta.mean())))
    pd.DataFrame(macro).to_csv(OUT/'reliability_repeats.csv',index=False)
    reliability=pd.DataFrame(records);reliability.to_csv(OUT/'reliability.csv',index=False)
    return curve,pd.DataFrame(rows),reliability


def plots(mechanism,validation,array,curve,reliability):
    plt.rcParams.update({'font.family':'serif','font.size':9,'axes.labelsize':9,'axes.titlesize':9,'legend.fontsize':7,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig,axs=plt.subplots(1,3,figsize=(7.1,2.15))
    ax=axs[0];ax.semilogx(mechanism.eta,mechanism.q_exact,label='MAP, exact')
    ax.semilogx(mechanism.eta[::3],mechanism.q_mc[::3],'o',ms=3,label='MAP, MC')
    ax.semilogx(mechanism.eta,mechanism.q_direct,'--',label='Direct report')
    ax.set(xlabel=r'Budget $\eta$',ylabel='Outage probability',title='(a) Empirical CSI model');ax.legend()
    ax=axs[1];ax.semilogx(mechanism.eta,mechanism.aoii,label='MAP')
    ax.semilogx(mechanism.eta,mechanism.aoii_direct,'--',label='Direct report')
    ax.set(xlabel=r'Budget $\eta$',ylabel=r'Mean AoII / $T$',title='(b) Privacy cost');ax.legend()
    ax=axs[2]
    for a,b in validation[validation.kind=='controlled'].groupby('alpha'):
        xx=np.linspace(.1,.9,200);line=ax.plot(xx,F(xx,a),label=rf'$\alpha={a:g}$')[0]
        ax.errorbar(b.q,b.simulation,yerr=b.halfwidth95,fmt='o',ms=3,color=line.get_color(),capsize=2)
    ax.set(xlabel='Outage probability',ylabel=r'Mean AoII / $T$',title='(c) Formula vs. slot simulation');ax.legend(ncol=2)
    for ax in axs:ax.grid(alpha=.2)
    fig.tight_layout(pad=.5);save(fig,'mechanism_theory')
    fig,axs=plt.subplots(1,2,figsize=(7.1,2.25))
    for ax,mode,title in zip(axs,['fixed_power','equal_aligned_snr'],['(a) Fixed transmit power','(b) Fixed aligned-SNR scale']):
        for m,b in array[array['mode']==mode].groupby('M'):
            ax.semilogx(b.eta,b.aoii,label=f'M={m}')
        ax.set(xlabel=r'Budget $\eta$',ylabel=r'Mean AoII / $T$',title=title);ax.grid(alpha=.2);ax.legend(ncol=2)
    fig.tight_layout(pad=.5);save(fig,'array_effect')
    fig,axs=plt.subplots(1,3,figsize=(7.1,2.2))
    ax=axs[0];ax.fill_between(curve.eta,curve.L,curve.U,alpha=.2,label='Selection bounds')
    ax.semilogx(curve.eta,curve.J_exact,label='Exact empirical model')
    ax.axhline(.05,ls='--',color='black',lw=.8,label=r'$D_{\max}/T=0.05$')
    ax.set(xlabel=r'Budget $\eta$',ylabel=r'Mean AoII / $T$',title='(a) Budget feasibility');ax.legend()
    ax=axs[1]
    for method,b in reliability.groupby('method'):
        ax.semilogx(b.Ns,b.violation_rate,'o-',ms=3,label=method.replace('_',' '))
    ax.set(xlabel=r'Selection samples $N_s$',ylabel='False-selection rate',title='(b) Constraint violations');ax.legend()
    ax=axs[2]
    for method,b in reliability.groupby('method'):
        ax.semilogx(b.Ns,b.mean_eta,'o-',ms=3,label=method.replace('_',' '))
    ax.set(xlabel=r'Selection samples $N_s$',ylabel=r'Mean selected budget $\eta$',title='(c) Cost of conservatism');ax.legend()
    for ax in axs:ax.grid(alpha=.2)
    fig.tight_layout(pad=.5);save(fig,'selection_reliability')


def save(fig,name):
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight',pad_inches=.025)
    fig.savefig(FIG/f'{name}.png',dpi=180,bbox_inches='tight',pad_inches=.025)
    plt.close(fig)


def checks():
    x=np.linspace(-100,100,10001)
    assert np.all(np.diff(projection_cdf(x,.1))>=0)
    sites=np.array([[-5,-5,25],[5,-15,25],[15,-25,25]],float)
    c=exact_confusion(sites,.1)
    np.testing.assert_allclose(c.sum(axis=1),1)
    assert np.all(c>=0)
    # Projection law agrees with Monte Carlo for nontrivial decision boundaries.
    rng=np.random.default_rng(22);n=200000
    u=rng.normal(size=(n,3));u/=np.linalg.norm(u,axis=1,keepdims=True)
    z=sites[1]+rng.gamma(3,10,n)[:,None]*u
    counts=np.bincount(cdist(z,sites).argmin(1),minlength=3)/n
    np.testing.assert_allclose(counts,c[1],atol=.004)
    # Conditional Rician CDF vs complex-normal projection samples.
    k=10;snr=.8;cor=3.
    v=np.sqrt(snr*k/(k+1)*cor)+np.sqrt(snr/(k+1))*(rng.normal(size=n)+1j*rng.normal(size=n))/np.sqrt(2)
    expected=ncx2.cdf(2/(snr/(k+1)),2,2*k*cor)
    assert abs(np.mean(abs(v)**2<1)-expected)<.004


def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True)
    checks();print('Physics/statistical checks passed',flush=True)
    d,c,g,w,o,conf,q,scale=model()
    print('Exact conditional empirical model ready',flush=True)
    mechanism=paired_mechanism(d,c,g,w,o,conf,q,scale)
    print('Measured mechanism complete',flush=True)
    validation=source_validation(q)
    print('29 source validation configurations complete',flush=True)
    array=array_experiment();print('Array controls complete',flush=True)
    curve,selected,reliability=selection(q);print('Selection and reliability complete',flush=True)
    plots(mechanism,validation,array,curve,reliability)
    CONFIG['elapsed_seconds']=time.perf_counter()-start
    CONFIG['runtime']={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__}
    CONFIG['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    CONFIG['scope']='Conditional empirical CSI bootstrap; separate ideal Rician array controls; no calibrated RF link budget.'
    (OUT/'config.json').write_text(json.dumps(CONFIG,indent=2),encoding='utf-8')
    print(selected.to_string(index=False),flush=True)
    print(reliability.to_string(index=False),flush=True)
    print('Max validation relative error',validation.relative_error.max(),flush=True)
    print('Results:',OUT,flush=True)


if __name__=='__main__':main()
