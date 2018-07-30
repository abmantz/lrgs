import numpy as np
import scipy.stats as st

class Chain(dict):
    def __init__(self, params, nmc, trace='bs'):
        params._init_chain(self, nmc, trace)
        self.params = params
        self.nmc = nmc
        self.trace = trace
    def run(self, fix='', mention_every=None):
        for i in range(self.nmc):
            try:
                self.params.update(fix)
                self.params._store(self, i)
                if mention_every is not None and (i+1) % mention_every == 0:
                    print 'Done with Gibbs iteration', i+1
            except KeyboardInterrupt:
                break
    def extract(self, field, *args):
        return self.__getattribute__(field).__getitem__(args)
    def to_dict(self, trace):
        cols = self.params._namecolumns(trace)
        f = {}
        for col in cols:
            w = col.split('_')
            k = w.pop(0)
            i = [int(j) for j in w]
            i.insert(0, k)
            f[col] = self.extract(*i)
        return f
    def to_recarray(self, trace):
        cols = self.params._namecolumns(trace)
        f = np.recarray((self.nmc, len(cols)), dtype=[(c, float) for c in cols])
        for col in cols:
            w = col.split('_')
            k = w.pop(0)
            i = [int(j) for j in w]
            i.insert(0, k)
            f[col] = np.asmatrix(self.extract(*i)).T
        return f
        

# base class; does not implement a p(x) model and hence cannot sample X
class Parameters:
    def __init__(self, xdata, ydata, M=None, M_inv=None, intercept=True, Sigma_prior=None):
        # todo: infinitely many checks
        # especially that things are/are not matrices
        self.n = ydata.shape[0]
        self.m = ydata.shape[1]
        if xdata is None:
            self.p = 0
            intercept = True # NB
        else: self.p = xdata.shape[1]
        self.pin = 0
        if intercept: self.pin = 1
        self.prange = np.arange(self.p) + self.pin
        self.xdata = xdata
        self.ydata = ydata
        if Sigma_prior is None:
            self.Sigma_prior_dof = -1
            self.Sigma_prior_scale = np.matrix(np.zeros((self.m, self.m)), copy=False)
        else:
            self.Sigma_prior_dof = Sigma_prior[0]
            self.Sigma_prior_scale = Sigma_prior[1]
        if M_inv is None:
            if M is None:
                self.M_inv = [np.asmatrix(np.eye(self.p+self.m)) for i in range(self.n)]
            else:
                self.M_inv = [np.linalg.inv(M[i]) for i in range(self.n)]
        else:
            self.M_inv = M_inv
        self.M = M
        # set initial values
        if intercept:
            if self.p == 0:
                self.X = np.matrix(np.ones(self.n), copy=False).transpose()
            else:
                self.X = np.concatenate((np.matrix(np.ones(self.n), copy=False).transpose(), xdata.copy()), axis=1) # n*(p+pin)
        else:
            self.X = xdata.copy()
        self.Y = ydata.copy()
        self.B = np.matrix(np.zeros((self.p+self.pin,self.m)), copy=False) # (p+pin)*m
        if intercept: self.B[0,:] = np.mean(self.Y, axis=0)
        self.Sigma = np.matrix(np.diag(np.var(self.Y, axis=0)), copy=True) # m*m
        self.Sigma_inv = 1.0 / self.Sigma
    def update_Sigma(self):
        E = self.Y - self.X * self.B # n*m
        self.Sigma_inv = st.wishart.rvs(self.n + self.Sigma_prior_dof, np.linalg.inv(E.T*E + self.Sigma_prior_scale))
        if self.m == 1:
            self.Sigma_inv = np.matrix(self.Sigma_inv)
            self.Sigma = 1.0 / self.Sigma_inv
        else:
            self.Sigma = np.linalg.inv(self.Sigma_inv)
    def update_B(self):
        Y_tilde = np.matrix(self.Y.flatten('F')).T # nm*1
        B_tilde_cal = np.zeros(self.m*(self.p+self.pin)) # (p+pin)m
        S_tilde_cal = np.matrix(np.zeros((self.m*(self.p+self.pin), self.m*(self.p+self.pin))), copy=False) # [(p+pin)m]^2
        XtXinv = np.linalg.inv(self.X.T * self.X) # (p+pin)^2
        XtXinvXt = XtXinv * self.X.T
        for j in range(self.m): B_tilde_cal[np.arange(self.p+self.pin)+j*(self.p+self.pin)] = (XtXinvXt * Y_tilde[np.arange(self.n)+j*self.n,0]).T # each chunk (p+pin)*1 is the solution (B) to X*B=Y
        for i in range(self.m):
            for j in range(self.m):
                S_tilde_cal[np.ix_(np.arange(self.p+self.pin)+i*(self.p+self.pin), np.arange(self.p+self.pin)+j*(self.p+self.pin))] = XtXinv * self.Sigma[i,j]
        #if Bprior:
        #    todo
        self.B = np.matrix(np.random.multivariate_normal(B_tilde_cal, S_tilde_cal))
        self.B.shape = (self.m, self.p+self.pin)
        self.B = self.B.T
    def update_Y(self):
        pred = self.X * self.B # n*m
        q = pred - self.Y # n*m
        if self.p == 0:
            zi = self.ydata - self.Y # n*(p+m)
        else :
            zi = np.concatenate((self.xdata-self.X[:,self.prange] , self.ydata-self.Y), axis=1) # NB skipping of intercept column in X, if any; n*(p+m)
        eta_hat = np.ndarray((self.n, self.m)) # n*m
        s2 = 1.0 / (np.array([np.asarray(self.M_inv[i]).diagonal()[self.p+np.arange(self.m)] for i in range(self.n)]) + np.outer(np.ones(self.n), np.asarray(self.Sigma_inv).diagonal())) # n*m
        for j in range(self.m):
            zi_star = zi.copy() # n*(p+m)
            zi_star[:,self.p+j] = self.ydata[:,j]
            qi_star = q.copy() # n*m
            qi_star[:,j] = pred[:,j]
            M_inv_rows = np.array([np.asarray(self.M_inv[i][self.p+j,:])[0] for i in range(self.n)])
            S_inv_rows = np.outer(np.ones(self.n), np.asarray(self.Sigma_inv[j,:])[0])
            eta_hat[:,j] = s2[:,j] * (np.sum(M_inv_rows*np.asarray(zi_star), axis=1) + np.sum(S_inv_rows*np.asarray(qi_star), axis=1))
        self.Y = np.asmatrix(np.random.normal(eta_hat, np.sqrt(s2)))
        # eta_hat = np.zeros(self.m)
        # for i in range(self.n):
        #     if self.p == 0:
        #         zi = self.ydata[i,:] - self.Y[i,:] # p+m
        #     else:
        #         zi = np.concatenate((self.xdata[i,:], self.ydata[i,:])) - np.concatenate((self.X[i,self.prange], self.Y[i,:])) # NB skipping of intercept column in X, if any; p+m
        #     s2 = 1.0/(self.M_inv[i].diagonal()[self.p+np.arange(self.m)] + self.Sigma_inv.diagonal()) # m
        #     for j in range(self.m):
        #         zi_star = zi.copy() # p+m
        #         zi_star[self.p+j] = self.ydata[i,j]
        #         qi_star = q[i,:] # m
        #         qi_star[j] = pred[i,j]
        #         eta_hat[j] = s2[j] * (np.dot(self.M_inv[i][self.p+j,:], zi_star) + np.dot(self.Sigma_inv[j,:], qi_star)) # scalar (index over m)
        #         # next response j
        #     self.Y[i,:] = np.random.normal(eta_hat, np.sqrt(s2))
        #     # next data point
    def update(self, fix=''):
        if fix.find('s') == -1: self.update_Sigma()
        if fix.find('b') == -1: self.update_B()
        if fix.find('y') == -1: self.update_Y()
    def _init_chain(self, chain, nmc, trace):
        if trace.find('s') != -1: chain.Sigma = np.full((self.m, self.m, nmc), np.nan)
        if trace.find('b') != -1: chain.B = np.full((self.p+self.pin, self.m, nmc), np.nan)
        if trace.find('y') != -1: chain.Y = np.full((self.n, self.m, nmc), np.nan)
    def _store(self, chain, i):
        if chain.trace.find('s') != -1: chain.Sigma[:,:,i] = self.Sigma
        if chain.trace.find('b') != -1: chain.B[:,:,i] = self.B
        if chain.trace.find('y') != -1: chain.Y[:,:,i] = self.Y
    def _namecolumns(self, trace):
        c = []
        for param in trace:
            if param == 'Sigma':
                for i in range(self.m):
                    for j in range(i+1):
                        c.append('_'.join(['Sigma', str(i), str(j)]))
            elif param == 'B':
                for i in range(self.pin+self.p):
                    for j in range(self.m):
                        c.append('_'.join(['B', str(i), str(j)]))
            elif param == 'Y':
                for i in range(self.n):
                    for j in range(self.m):
                        c.append('_'.join(['Y', str(i), str(j)]))
        return c



class Parameters_Uniform(Parameters):
    def update_X(self):
        B_Sinv = self.B[self.prange,:] * self.Sigma_inv # p*m
        B_Sinv_B_j = np.zeros(self.p)
        for j in range(self.p):
            B_Sinv_B_j[j] = np.dot(B_Sinv[j,:], self.B[self.pin+j,:])
        zi = zi = np.concatenate((self.xdata-self.X[:,self.prange] , self.ydata-self.Y), axis=1) # NB skipping of intercept column in X, if any; n*(p+m)
        s2 = 1.0 / (np.array([np.asarray(self.M_inv[i]).diagonal()[np.arange(self.p)] for i in range(self.n)]) + np.outer(np.ones(self.n), np.asarray(B_Sinv_B_j))) # n*p
        xi_hat = np.ndarray((self.n,self.p))
        for j in range(self.p):
            zi_star = zi.copy() # n*(p+m)
            zi_star[:,j] = self.xdata[:,j]
            inds = range(self.pin+self.p)
            inds.pop(self.pin+j)
            pred = np.array([np.asarray(self.X[i,inds] * self.B[inds,:])[0] for i in range(self.n)]) # n*m; not seeing a simple way to avoid the loop here...
            M_inv_rows = np.array([np.asarray(self.M_inv[i][j,:])[0] for i in range(self.n)])
            B_Sinv_rows = np.outer(np.ones(self.n), np.asarray(B_Sinv[j,:])[0])
            xi_hat[:,j] = s2[:,j] * (np.sum(M_inv_rows*np.asarray(zi_star), axis=1) + np.sum(B_Sinv_rows*np.asarray(self.Y-pred), axis=1))
        self.X[:,self.prange] = np.random.normal(xi_hat, np.sqrt(s2))
        # xi_hat = np.zeros(self.p)
        # s2 = np.zeros(self.p)
        # for i in range(self.n):
        #   zi = np.concatenate((self.xdata[i,:], self.ydata[i,:])) - np.concatenate((self.X[i,self.prange], self.Y[i,:])) # p+m
        #   for j in range(self.p):
        #     s2[j] = 1.0/(self.M_inv[i][j,j] + B_Sinv_B_j[j])
        #     zi_star = zi.copy() # p+m
        #     zi_star[j] = self.xdata[i,j]
        #     inds = range(self.pin+self.p)
        #     inds.pop(self.pin+j)
        #     pred = self.X[i,inds] * self.B[inds,:] # 1*m
        #     xi_hat[j] = s2[j] * (np.dot(self.M_inv[i][j,:], zi_star) + np.dot(B_Sinv[j,:], self.Y[i,:] - pred))
        #     # next covariate
        #   self.X[i,self.prange] = np.random.normal(xi_hat, np.sqrt(s2))
        #   # next data point
    def update(self, fix=''):
        Parameters.update(self, fix)
        if fix.find('x') == -1: self.update_X()
    def _init_chain(self, chain, nmc, trace):
        Parameters._init_chain(self, chain, nmc, trace)
        if trace.find('x') != -1: chain.X = np.full((self.n, self.p, nmc), np.nan)
    def _store(self, chain, i):
        Parameters._store(self, chain, i)
        if chain.trace.find('x') != -1:
            if self.pin == 1: chain.X[:,:,i] = self.X[:,-1]
            else: chain.X[:,:,i] = self.X
    def _namecolumns(self, trace):
        c = Parameters._namecolumns(self, trace)
        for param in trace:
            if param == 'X':
                for i in range(self.n):
                    for j in range(self.m):
                        c.append('_'.join(['X', str(i), str(j)]))
        return c



class Parameters_GaussMix(Parameters):
    def __init__(self, Ngauss, *args, **kwargs):
        Parameters.__init__(self, *args, **kwargs)
        self.Ngauss = Ngauss
        # set initial values
        self.G = np.random.choice(range(Ngauss), self.n) # n
        self.nG = np.array([len(np.where(self.G == k)[0]) for k in range(Ngauss)]) # Ngauss
        self.pi = (1.0*self.nG) / self.n # Ngauss
        self.mu0 = np.mean(self.xdata, axis=0).T # p
        self.U = np.matrix(np.diag(np.var(self.xdata, axis=0)), copy=False) # p*p
        self.U_inv = np.linalg.inv(self.U)
        self.mu = [np.matrix(np.random.multivariate_normal(np.array(self.mu0)[:,0], self.U)).T for k in range(Ngauss)] # Ngauss of p
        self.W = np.matrix(np.cov(self.xdata.T)) # p*p
        self.Tau = [np.matrix(st.invwishart.rvs(Ngauss+self.p, self.W)) for k in range(Ngauss)] # Ngauss of p*p
        self.Tau_inv = [np.linalg.inv(self.Tau[k]) for k in range(Ngauss)]
        if Ngauss > 1: self.nu_Tau = self.p
        else:
            self.nu_Tau = 0
            self.U_inv *= 0.0
            self.W *= 0.0
    def update_Tau(self):
        for k in range(self.Ngauss):
            ii = np.where(self.G == k)[0]
            S = self.W.copy()
            for i in ii:
                z = self.X[i, self.prange] - self.mu[k].T
                S += np.outer(z, z)
            self.Tau_inv[k] = np.matrix(st.wishart.rvs(len(ii) + self.nu_Tau, np.linalg.inv(S)))
            self.Tau[k] = np.linalg.inv(self.Tau_inv[k])
    def update_pi(self):
        if self.Ngauss > 1: self.pi = np.random.dirichlet(1 + self.nG)
    def update_X(self):
        B_Sinv = self.B[self.prange,:] * self.Sigma_inv # p*m
        B_Sinv_B_j = np.array([np.dot(B_Sinv[j,:], self.B[self.pin+j,:]) for j in range(self.p)])
        zi = zi = np.concatenate((self.xdata-self.X[:,self.prange] , self.ydata-self.Y), axis=1) # NB skipping of intercept column in X, if any; n*(p+m)
        mui = np.array([np.asarray(self.mu[self.G[i]].T)[0] for i in range(self.n)]) - self.X[i,self.prange] # n*p
        s2 = 1.0 / (np.array([np.asarray(self.M_inv[i]).diagonal()[np.arange(self.p)] for i in range(self.n)]) + np.outer(np.ones(self.n), np.asarray(B_Sinv_B_j))) # n*p
        xi_hat = np.ndarray((self.n,self.p))
        for j in range(self.p):
            zi_star = zi.copy() # n*(p+m)
            zi_star[:,j] = self.xdata[:,j]
            mui_star = mui.copy() # n*p
            mui_star[:,j] = np.array([[self.mu[self.G[i]][j,0]] for i in range(self.n)])
            inds = range(self.pin+self.p)
            inds.pop(self.pin+j)
            pred = np.array([np.asarray(self.X[i,inds] * self.B[inds,:])[0] for i in range(self.n)]) # n*m; not seeing a simple way to avoid the loop here...
            M_inv_rows = np.array([np.asarray(self.M_inv[i][j,:])[0] for i in range(self.n)])
            Tau_inv_rows = np.array([np.asarray(self.Tau_inv[self.G[i]][j,:])[0] for i in range(self.n)])
            B_Sinv_rows = np.outer(np.ones(self.n), np.asarray(B_Sinv[j,:])[0])
            xi_hat[:,j] = s2[:,j] * (np.sum(M_inv_rows*np.asarray(zi_star), axis=1) + np.sum(Tau_inv_rows*np.asarray(mui_star), axis=1) + np.sum(B_Sinv_rows*np.asarray(self.Y-pred), axis=1))
        self.X[:,self.prange] = np.random.normal(xi_hat, np.sqrt(s2))
        # xi_hat = np.zeros(self.p)
        # s2 = np.zeros(self.p)
        # for i in range(self.n):
        #   zi = np.concatenate((self.xdata[i,:], self.ydata[i,:])) - np.concatenate((self.X[i,self.prange], self.Y[i,:])) # p+m
        #   mui = self.mu[self.G[i]].T - self.X[i,self.prange] # p
        #   for j in range(self.p):
        #     s2[j] = 1.0/(self.M_inv[i][j,j] + self.Tau_inv[self.G[i]][j,j] + B_Sinv_B_j[j])
        #     zi_star = zi.copy() # p+m
        #     zi_star[j] = self.xdata[i,j]
        #     mui_star = mui.copy() # p
        #     mui_star[j] = self.mu[self.G[i]][j,0]
        #     inds = range(self.pin+self.p)
        #     inds.pop(self.pin+j)
        #     pred = self.X[i,inds] * self.B[inds,:] # 1*m
        #     xi_hat[j] = s2[j] * (np.dot(self.M_inv[i][j,:], zi_star) + np.dot(self.Tau_inv[self.G[i]][j,:], mui_star) + np.dot(B_Sinv[j,:], self.Y[i,:] - pred))
        #     # next covariate
        #   self.X[i,self.prange] = np.random.normal(xi_hat, np.sqrt(s2))
        #   # next data point
    def update_G(self):
        if self.Ngauss < 2:
            return
        for i in range(self.n):
            q = np.array([self.pi[k] * st.multivariate_normal.pdf(self.X[i,self.prange], self.mu[k], self.Tau[k]) for k in range(self.Ngauss)])
            q /= np.sum(q)
            self.G[i] = np.where(np.random.multinomial(1, q) == 1)[0][0]
        self.nG = np.array([len(np.where(self.G == k)[0]) for k in range(self.Ngauss)])
    def update_mu(self):
        for k in range(self.Ngauss):
            nk = self.nG[k]
            if nk == 0:
                S_mu = self.U
                mu_hat = self.mu0
            else:
                gg = np.where(self.G == k)[0]
                S_mu = np.linalg.inv(self.U_inv + nk*self.Tau_inv[k])
                mu_hat = S_mu * (self.U_inv * self.mu0 + nk * self.Tau_inv[k] * np.mean(self.X[np.ix_(gg, self.prange)], axis=0))
            self.mu[k] = np.matrix(np.random.multivariate_normal(np.array(mu_hat.T)[0,:], S_mu)).T
    def update_mu0(self):
        if self.Ngauss < 2:
            return
        m = np.array([np.mean([self.mu[k][j] for k in range(self.Ngauss)]) for j in range(self.p)])
        self.mu0 = np.matrix(np.random.multivariate_normal(m, self.U/(1.0*self.Ngauss))).T
    def update_U(self):
        if self.Ngauss < 2:
            return
        S = self.W.copy()
        for k in range(self.Ngauss):
            z = self.mu[k] - self.mu0
            S += z * z.T
        self.U_inv = np.matrix(st.wishart.rvs(self.Ngauss + self.p, np.linalg.inv(S)))
        self.U = np.linalg.inv(self.U_inv)
    def update_W(self):
        if self.Ngauss < 2:
            return
        S = self.U_inv.copy()
        for k in range(self.Ngauss): S += self.Tau_inv[k]
        self.W = np.matrix(st.wishart.rvs((self.Ngauss + 2)*self.p + 1, np.linalg.inv(S)))
    def update(self, fix=''):
        Parameters.update(self, fix)
        if fix.find('t') == -1: self.update_Tau()
        if fix.find('p') == -1: self.update_pi()
        if fix.find('x') == -1: self.update_X()
        if fix.find('g') == -1: self.update_G()
        if fix.find('m') == -1: self.update_mu()
        if fix.find('z') == -1: self.update_mu0()
        if fix.find('u') == -1: self.update_U()
        if fix.find('w') == -1: self.update_W()
    def _init_chain(self, chain, nmc, trace):
        Parameters._init_chain(self, chain, nmc, trace)
        if trace.find('x') != -1: chain.X = np.full((self.n, self.p, nmc), np.nan)
        if trace.find('t') != -1: chain.Tau = np.full((self.Ngauss, self.p, self.p, nmc), np.nan)
        if trace.find('p') != -1: chain.pi = np.full((self.Ngauss, nmc), np.nan)
        if trace.find('g') != -1: chain.G = np.full((self.n, nmc), np.nan)
        if trace.find('m') != -1: chain.mu = np.full((self.Ngauss, self.p, nmc), np.nan)
        if trace.find('z') != -1: chain.mu0 = np.full((self.p, nmc), np.nan)
        if trace.find('u') != -1: chain.U = np.full((self.p, self.p, nmc), np.nan)
        if trace.find('w') != -1: chain.W = np.full((self.p, self.p, nmc), np.nan)
    def _store(self, chain, i):
        Parameters._store(self, chain, i)
        if chain.trace.find('x') != -1:
            if self.pin == 1: chain.X[:,:,i] = self.X[:,-1]
            else: chain.X[:,:,i] = self.X
        if chain.trace.find('t') != -1: chain.Tau[:,:,:,i] = np.array(self.Tau)
        if chain.trace.find('p') != -1: chain.pi[:,i] = self.pi
        if chain.trace.find('g') != -1: chain.G[:,i] = self.G
        if chain.trace.find('m') != -1: chain.mu[:,:,i] = np.array([np.array(self.mu[k].T)[0] for k in range(self.Ngauss)])
        if chain.trace.find('z') != -1: chain.mu0[:,i] = self.mu0
        if chain.trace.find('u') != -1: chain.U[:,:,i] = self.U
        if chain.trace.find('w') != -1: chain.W[:,:,i] = self.W
    def _namecolumns(self, trace):
        c = Parameters._namecolumns(self, trace)
        for param in trace:
            if param == 'X':
                for i in range(self.n):
                    for j in range(self.m):
                        c.append('_'.join(['Y', str(i), str(j)]))
            elif param == 'Tau':
                for k in range(self.Ngauss):
                    for i in range(self.p):
                        for j in range(i+1):
                            c.append('_'.join(['Tau', str(k), str(i), str(j)]))
            elif param == 'pi':
                for k in range(self.Ngauss):
                    c.append('_'.join(['pi', str(k)]))
            elif param == 'G':
                for k in range(self.n):
                    c.append('_'.join(['G', str(k)]))
            elif param == 'mu':
                for k in range(self.Ngauss):
                    for j in range(self.p):
                        c.append('_'.join(['mu', str(k), str(j)]))
            elif param == 'mu0':
                for k in range(self.p):
                    c.append('_'.join(['mu0', str(k)]))
            elif param == 'U':
                for i in range(self.p):
                    for j in range(i+1):
                        c.append('_'.join(['U', str(i), str(j)]))
            elif param == 'W':
                for i in range(self.p):
                    for j in range(i+1):
                        c.append('_'.join(['W', str(i), str(j)]))
        return c

