import sys, numpy as np
sys.path.insert(0,'/Users/apunt/repos/NoiseInject')
from noiseInject.core import NoiseInjectorRegression as New
rs=np.random.RandomState(0); y=rs.normal(7,1.2,500)
for s in ['legacy','quantile','threshold','outlier','hetero','valprop']:
    n1=New(strategy=s,random_state=7); n2=New(strategy=s,random_state=7)
    for sig in [0.1,0.3,0.9]:
        a=n1.inject(y,sig); b,si,eps=n2.inject_verbose(y,sig)
        print(s,sig,'y_noisy equal:',np.array_equal(a,b),' eps roundtrip:',np.array_equal(b-y,eps))
