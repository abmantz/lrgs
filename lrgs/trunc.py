import lmc
import numpy as np
import scipy.stats as st

def nopriors(s):
    return 0.0

class GibbsUpdater:
    def __init__(self):
        self.engine = None
        self.index = None
        self.count = 0
        self.rate = 1
    def __call__(self, s):
        s.gibbs_sample()
        s.lnP = s(s) # hang on to this so that we can tell whether the non-Gibbs parameters change later
        self.engine.current_logP = s.lnP

class DerivedUpdater:
    def __init__(self):
        self.engine = None
        self.index = None
        self.count = 0
        self.rate = 1
    def __call__(self, s):
        if self.engine.current_logP != s.lnP:
            # Metropolis updater must have accepted its proposal
            # Update current values of derived parameters to be written to the chain
            s.fdet.set(s.fdet_tmp)


class PoissonTruncated: # should not be directly instantiated
    def __init__(self, par, fdet_func, N_prior=(0.5,0.0), extra_priors=nopriors, **kwargs):
        self.extra_priors = extra_priors
        self.fdet_func = fdet_func
        self.par = par
        self.N_prior_alpha = N_prior[0]
        self.N_prior_beta = N_prior[1]
        if par.M is None:
            par.M = [np.linalg.inv(par.M_inv[i]) for i in range(par.n)]
        # set up LMC parameter objects
        # todo: sensible width guesses? or make the users problem... or use GW (problematic)?
        self.space = lmc.ParameterSpace([], self)
        self.B = [[lmc.Parameter(par.B[i,j], 0.01, '_'.join(['B',str(i),str(j)])) for j in range(par.B.shape[1])] for i in range(par.B.shape[0])]
        for i in range(par.B.shape[0]):
                for j in range(par.B.shape[1]):
                        self.space.append(self.B[i][j])
        self.Sigma = [[lmc.Parameter(par.Sigma[i,j], 0.01, '_'.join(['Sigma',str(i),str(j)])) for j in range(i+1)] for i in range(par.Sigma.shape[0])]
        for i in range(par.Sigma.shape[0]):
                for j in range(i+1):
                        self.space.append(self.Sigma[i][j])
        self.fdet = lmc.Parameter(1.0, 0.01, 'fdet') # NOT a free parameter
        # derived classes must call finish_init after this
    def finish_init(self, step=lmc.Slice(), parallel=None, updater=lmc.MultiDimRotationUpdater, adapt_every=100, adapt_starting=100, **kwargs):
        self.trace = [p for p in self.space]
        self.trace.append(self.fdet)
        self.updater = updater(self.space, step, adapt_every, adapt_starting, parallel=parallel)
        self.engine = lmc.Engine([GibbsUpdater(), self.updater, DerivedUpdater()], self.trace)
    def __call__(self, s): # When called, s will be self, just for maximal confusion
        # Logically, most of this belongs in the lrgs::Parameter subclasses, but then
        # we would have to deal with reverting things if a proposal was rejected.
        Sigma = np.asmatrix(np.zeros(s.par.Sigma.shape))
        for i in range(s.par.m):
            Sigma[i,i] = s.Sigma[i][i]()
            for j in range(i):
                Sigma[i,j] = s.Sigma[i][j]()
                Sigma[j,i] = Sigma[i,j]
        det_Sigma = np.linalg.det(Sigma)
        if det_Sigma <= 0.0:
            return -np.inf
        lnp  = s.extra_priors(s)
        if not np.isfinite(lnp):
            return lnp
        lnp += -np.log(det_Sigma) # jeffreys prior on sigma - todo: allow invwish
        fdet = s.fdet_func(s)
        lnp += -(s.par.n + s.N_prior_alpha)*np.log(fdet + s.N_prior_beta) # fdet contribution: Poisson p(N|<N>), marginalized over <N>
        xtrue = self.par.X[:,self.par.pin+np.arange(self.par.p)]
        B = np.matrix([[s.B[i][j]() for j in range(s.par.B.shape[1])] for i in range(s.par.B.shape[0])])
        pred = self.par.X*B
        for i in range(s.par.n):
            lnp += st.multivariate_normal.logpdf(np.asarray(s.par.Y)[i,:], np.asarray(pred)[i,:], Sigma) # p(y|x,alpha,beta,sigma)
            lnp += st.multivariate_normal.logpdf(np.asarray(np.concatenate((s.par.xdata[i,:], s.par.ydata[i,:]))).reshape(s.par.p+s.par.m), np.asarray(np.concatenate((np.asarray(xtrue)[i,:], np.asarray(s.par.Y)[i,:]))).reshape(s.par.p+s.par.m), s.par.M[i]) # p(xhat,yhat|x,y)
        s.fdet_tmp = fdet
        return lnp
    def gibbs_sample(self):
        self.par.B = np.matrix([[self.B[i][j]() for j in range(self.par.B.shape[1])] for i in range(self.par.B.shape[0])])
        self.par.Sigma = np.asmatrix(np.zeros(self.par.Sigma.shape))
        for i in range(self.par.m):
            self.par.Sigma[i,i] = self.Sigma[i][i]()
            for j in range(i):
                self.par.Sigma[i,j] = self.Sigma[i][j]()
                self.par.Sigma[j,i] = self.par.Sigma[i,j]
        self.par.Sigma_inv = np.linalg.inv(self.par.Sigma)
        # derived classes must call the actual update
    def run(self, niter, backends=[lmc.stdoutBackend()]):
        self.engine(niter, self, backends)





class PoissonTruncated_GaussMix(PoissonTruncated):
    def __init__(self, *args, **kwargs):
        PoissonTruncated.__init__(self, *args, **kwargs)
        self.mu = [[lmc.Parameter(self.par.mu[k][j,0], 0.01, '_'.join(['mu',str(k),str(j)])) for j in range(self.par.p)] for k in range(self.par.Ngauss)]
        for k in range(self.par.Ngauss):
            for j in range(self.par.p):
                self.space.append(self.mu[k][j])
        self.Tau = [[[lmc.Parameter(self.par.Tau[k][i,j], 0.01, '_'.join(['Tau',str(i),str(j)])) for j in range(i+1)] for i in range(self.par.p)] for k in range(self.par.Ngauss)]
        for k in range(self.par.Ngauss):
            for i in range(self.par.p):
                for j in range(i+1):
                    self.space.append(self.Tau[k][i][j])
        self.pi = [lmc.Parameter(self.par.pi[k], 0.01, 'pi_'+str(k)) for k in range(self.par.Ngauss)] # last pi is NOT a free parameter
        for k in range(self.par.Ngauss-1):
            self.space.append(self.pi[k])
        self.finish_init(**kwargs)
    def __call__(self, s):
        lnp = PoissonTruncated.__call__(self, s)
        if not np.isfinite(lnp):
            return lnp
        if s.par.Ngauss > 1:
            s.pi[s.par.Ngauss-1].set(1.0 - np.sum([s.pi[k]() for k in range(s.par.Ngauss-1)]))
            if np.any(np.array([s.pi[k]() for k in range(s.par.Ngauss)]) <= 0.0):
                return -np.inf
        Tau = [np.asmatrix(np.zeros((s.par.p,s.par.p))) for k in range(s.par.Ngauss)]
        det_Tau = np.zeros(s.par.Ngauss)
        mu = []
        for k in range(s.par.Ngauss):
            for i in range(s.par.p):
                for j in range(s.par.p):
                    Tau[k][i,j] = s.Tau[k][i][np.min([i,j])]()
            det_Tau[k] = np.linalg.det(Tau[k])
            if det_Tau[k] <= 0.0:
                return -np.inf
            mu.append(np.array([s.mu[k][j]() for j in range(s.par.p)]))
        if s.par.Ngauss == 1:
            # W, mu0 and pi are not varied
            lnp += -np.log(det_Tau[0]) # jeffreys prior on tau and mu (uniform)
        else: # Ngauss > 1
            for k in range(s.par.Ngauss):
                lnp += st.invwishart.logpdf(Tau[k], s.par.Ngauss+s.par.p, s.par.W) # p(Tau|W)
                lnp += st.multivariate_normal.logpdf(mu[k], s.par.mu0, s.par.U) # p(mu|mu0,U)
            lnp += st.dirichlet.logpdf(np.array([s.pi[k]() for k in range(s.par.Ngauss)]), [1.0]*s.par.Ngauss) # p(pi)
        xtrue = self.par.X[:,self.par.pin+np.arange(self.par.p)]
        for i in range(s.par.n):
            lnp += st.multivariate_normal.logpdf(xtrue[i,:], mu[s.par.G[i]], Tau[s.par.G[i]]) # p(x|mu,Tau)   (could have 1 call per Gaussian component)
        return lnp
    def gibbs_sample(self):
        PoissonTruncated.gibbs_sample(self)
        if self.par.Ngauss > 1:
            self.pi[self.par.Ngauss-1].set(1.0 - np.sum([self.pi[k]() for k in range(self.par.Ngauss-1)]))
        for k in range(self.par.Ngauss):
            if self.par.Ngauss > 1:
                self.par.pi[k] = self.pi[k]()
            self.par.mu[k] = np.matrix([self.mu[k][j]() for j in range(self.par.p)]).T
            for i in range(self.par.p):
                for j in range(self.par.p):
                    self.par.Tau[k][i,j] = self.Tau[k][i][np.min([i,j])]()
            self.par.Tau_inv[k] = np.linalg.inv(self.par.Tau[k])
        self.par.update(fix='bspmt')



class PoissonTruncated_ExpMix(PoissonTruncated):
    def __init__(self, *args, **kwargs):
        PoissonTruncated.__init__(self, *args, **kwargs)
        self.rate = [[lmc.Parameter(self.par.rate[k][j], 0.01, '_'.join(['rate',str(k),str(j)])) for j in range(self.par.p)] for k in range(self.par.Nexp)]
        for k in range(self.par.Nexp):
            for j in range(self.par.p):
                self.space.append(self.rate[k][j])
        self.pi = [lmc.Parameter(self.par.pi[k], 0.01, 'pi_'+str(k)) for k in range(self.par.Nexp)] # last pi is NOT a free parameter
        for k in range(self.par.Nexp-1):
            self.space.append(self.pi[k])
        self.finish_init(**kwargs)
    def __call__(self, s):
        lnp = PoissonTruncated.__call__(self, s)
        if not np.isfinite(lnp):
            return lnp
        if s.par.Nexp > 1:
            s.pi[s.par.Nexp-1].set(1.0 - np.sum([s.pi[k]() for k in range(s.par.Nexp-1)]))
            if np.any(np.array([s.pi[k]() for k in range(s.par.Nexp)]) <= 0.0):
                return -np.inf
        rate = []
        for k in range(s.par.Nexp):
            rate.append(np.array([s.rate[k][j]() for j in range(s.par.p)]))
        # no fancy hyperpriors on rates for the moment
        if s.par.Nexp == 1:
            pass
        else: # Nexp > 1
            lnp += st.dirichlet.logpdf(np.array([s.pi[k]() for k in range(s.par.Nexp)]), [1.0]*s.par.Nexp) # p(pi)
        xtrue = self.par.X[:,self.par.pin+np.arange(self.par.p)]
        rates = np.array([rate[s.par.G[i]] for i in range(s.par.n)])
        lnp += np.sum(st.expon.logpdf(xtrue, scale=1.0/rates))
        return lnp
    def gibbs_sample(self):
        PoissonTruncated.gibbs_sample(self)
        if self.par.Nexp > 1:
            self.pi[self.par.Nexp-1].set(1.0 - np.sum([self.pi[k]() for k in range(self.par.Nexp-1)]))
        for k in range(self.par.Nexp):
            if self.par.Nexp > 1:
                self.par.pi[k] = self.pi[k]()
            self.par.rate[k] = np.array([self.rate[k][j]() for j in range(self.par.p)])
        self.par.update(fix='bspl')
