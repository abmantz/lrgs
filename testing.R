library(rstan)
options(mc.cores = parallel::detectCores())

data <- read.table("toy.dat",header=TRUE)
M <- replicate(nrow(data),list(diag(2)))
lrgs.data <- list(Ngauss =3,
                  n = nrow(data),
                  x = data$x,
                  y = data$y,
                  M = M)

Sys.setenv(R_MAKEVARS_USER=file.path(getwd(),"stan_makevars"))
sm <- stan_model("p1_m1_Gmix.stan")

fit <- sampling(sm,data=lrgs.data,iter=4000,chains=2, control=list(max_treedepth=10))

fit <- sampling(sm,data=lrgs.data,iter=4000,chains=2, control=list(max_treedepth=10,adapt_delta=0.9))

pairs(fit,pars=c("alpha","beta","Sigma"))
traceplot(fit,pars=c("alpha","beta","Sigma","lp__"),inc_warmup=TRUE)
traceplot(fit,pars=c("pi","mu","Tau"),inc_warmup=FALSE)
traceplot(fit,pars=c("mu0","mu"),inc_warmup=FALSE)
traceplot(fit,pars=c("mu0","Tau","U","W","lp__"))
       
traceplot(fit,pars=c("alpha","beta","Sigma","mu_raw","mu0","log_Tau","log_U","log_W"))
pairs(fit,pars=c("alpha","beta","Sigma","mu_raw","mu0","log_Tau","log_U","log_W"))

pairs(fit,pars=c("alpha","beta","Sigma","mu_raw","mu0","log_Tau_raw","log_U_raw","log_W_raw",'xi[1]'))
