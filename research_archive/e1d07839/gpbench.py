import time, numpy as np, torch, gpytorch, warnings
warnings.filterwarnings('ignore')
try:
    from botorch.fit import fit_gpytorch_mll as fitm
except ImportError:
    from botorch import fit_gpytorch_model as fitm
torch.set_num_threads(8)
class G(gpytorch.models.ExactGP):
    def __init__(s,x,y,l):
        super().__init__(x,y,l); s.mean_module=gpytorch.means.ConstantMean()
        s.covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(s,x): return gpytorch.distributions.MultivariateNormal(s.mean_module(x),s.covar_module(x))
def run(n,d,seed=0):
    rng=np.random.default_rng(seed); X=rng.normal(size=(n,d)); y=rng.normal(size=n)
    Xt=torch.from_numpy(X); yt=torch.from_numpy(y)
    lik=gpytorch.likelihoods.GaussianLikelihood(noise=1e-3)
    m=G(Xt,yt,lik); m.covar_module.outputscale=1.0
    mll=gpytorch.mlls.ExactMarginalLogLikelihood(lik,m)
    t=time.time(); fitm(mll); return time.time()-t
for n,d in [(1000,200),(1000,2048),(2000,200),(2000,2048)]:
    try:
        print(f"n={n} d={d}: {run(n,d):.1f}s", flush=True)
    except Exception as e:
        print(n,d,"ERR",e, flush=True)
