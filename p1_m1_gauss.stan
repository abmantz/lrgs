data {
    int<lower=1> n;
    vector[n] x;
    vector[n] y;
    matrix[2,2] M[n];
}
transformed data {
    vector[2] xy[n];
    for (i in 1:n) {
        xy[i][1] = x[i];
        xy[i][2] = y[i];
    }
}
parameters {
    vector[n] xi;
    vector[n] eta;
    real alpha;
    real beta;
    real<lower=0> Sigma;
    real mu;
    real<lower=0> Tau;
}
transformed parameters {
    vector[2] xieta[n];
    for (i in 1:n) {
        xieta[i][1] = xi[i];
        xieta[i][2] = eta[i];
    }
}
model {
    Sigma ~ inv_gamma(1e-3, 1e-3);
    Tau ~ inv_gamma(1e-3, 1e-3);
    xi ~ normal(mu, sqrt(Tau));
    eta ~ normal(alpha+beta*x, sqrt(Sigma));
    for (i in 1:n)
        xy[i] ~ multi_normal(xieta[i], M[i]);
}
