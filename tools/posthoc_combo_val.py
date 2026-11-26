import sys; sys.argv=['x','logs/CARGO_pilot/personvit_experts4']; sys.path.insert(0,'tools'); sys.path.insert(0,'.')
import numpy as np, os.path as osp
import posthoc_val as ph
for name in ['CARGO_VAL','CARGO_VAL_AG','CARGO_VAL_AA','CARGO_VAL_GG']:
    d=np.load(f'logs/posthoc/personvit_experts4_{name}.npz'); F,G,P,C,nq=d['F'],d['G'],d['P'],d['C'],int(d['nq'])
    nl=np.array(['night' in osp.basename(p) for p in d['paths']])
    fl=ph.l2n(F+G)
    V={'base':F,'aqe k2 a3':ph.aqe(F,nq,2,3.),'flip+aqe k2 a3':ph.aqe(fl,nq,2,3.),
       'flip+aqe k3 a3':ph.aqe(fl,nq,3,3.),'cam.5+aqe k2 a3':ph.aqe(ph.center(F,C,.5),nq,2,3.),
       'flip+cam.5+aqe k2 a3':ph.aqe(ph.center(fl,C,.5),nq,2,3.),'aqe k5 a3':ph.aqe(F,nq,5,3.)}
    print('==',name)
    for v,X in V.items():
        r=ph.evaluate(X,P,C,nq,nl); print(f"{v:24s} mAP {r['mAP']:6.2f} R1 {r['R1']:6.2f} night {r['night_mAP']:6.2f}/{r['night_R1']:6.2f}")
