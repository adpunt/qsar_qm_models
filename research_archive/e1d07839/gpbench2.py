import time, numpy as np, torch, gpytorch, warnings
warnings.filterwarnings('ignore')
torch.set_num_threads(8)
class G(gpytorch.models.ExactGP):
    def __init__(s,x,y,l):
        super().__init__(x,y,l); s.mean_module=gpytorch.means.ConstantMean()
        s.covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(s,x): return gpytorch.distributions.MultivariateNormal(s.mean_module(x),s.covar_module(x))
def run(n,d,iters=5,seed=0):
    rng=np.random.default_rng(seed); X=rng.normal(size=(n,d)); y=rng.normal(size=n)
    Xt=torch.from_numpy(X); yt=torch.from_numpy(y)
    lik=gpytorch.likelihoods.GaussianLikelihood(noise=1e-3)
    m=G(Xt,yt,lik); m.covar_module.outputscale=1.0
    mll=gpytorch.mlls.ExactMarginalLogLikelihood(lik,m)
    opt=torch.optim.Adam(m.parameters(),lr=0.1); m.train(); lik.train()
    t=time.time()
    for _ in range(iters):
        opt.zero_grad(); out=m(Xt); loss=-mll(out,yt); loss.backward(); opt.step()
    return (time.time()-t)/iters
for n,d in [(2000,200),(2000,2048),(1600,200),(1600,2048)]:
    print(f"n={n} d={d}: {run(n,d):.3f}s/iter", flush=True)
