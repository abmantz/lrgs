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
    eta ~ normal(alpha+beta*x, sqrt(Sigma));
    //for (i in 1:n) {
    //    xi[i] ~ normal(x[i], sqrt(M[i][1,1]));
    //    eta[i] ~ normal(y[i], sqrt(M[i][2,2]));
    //}
    for (i in 1:n)
        xy[i] ~ multi_normal(xieta[i], M[i]);
}
