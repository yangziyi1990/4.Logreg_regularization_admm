% Distributed Lhalf regularized logistic regression
% (compared against lhalf_logreg package)

% Generate problem data

rand('seed', 0);
randn('seed', 0);

n = 400;  % the number of samples;
p = 200;  % the number of features
N = 10;  % the number of agent;

beta_int = sprandn(p, 1, 100/p);       % N(0,1), 10% sparse
beta_zero = randn(1);                  % random intercept

X0 = sprandn(n*N, p, 10/p);           % data / observations
Ytrue = sign(X0*beta_int + beta_zero);

% noise is function of problem size use 0.1 for large problem
Y0 = sign(X0*beta_int + beta_zero + sqrt(0.1)*randn(n*N, 1)); % labels with noise

% packs all observations in to an m*N x n matrix
A0 = spdiags(Y0, 0, n*N, n*N) * X0;

ratio = sum(Y0 == 1)/(n*N);
lambda = 0.1*1/(n*N) * norm((1-ratio)*sum(A0(Y0==1,:),1) + ratio*sum(A0(Y0==-1,:),1), 'inf');

beta_true = [beta_zero; beta_int];

% Solve problem

[beta history] = distr_lhalf_logreg(A0, Y0, lambda, N, 1.0, 1.0);

% Reporting

K = length(history.objval);

h = figure;
plot(1:K, history.objval, 'k', 'MarkerSize', 10, 'LineWidth', 2);
ylabel('f(x^k) + g(z^k)'); xlabel('iter (k)');

g = figure;
subplot(2,1,1);
semilogy(1:K, max(1e-8, history.r_norm), 'k', ...
    1:K, history.eps_pri, 'k--',  'LineWidth', 2);
ylabel('||r||_2');

subplot(2,1,2);
semilogy(1:K, max(1e-8, history.s_norm), 'k', ...
    1:K, history.eps_dual, 'k--', 'LineWidth', 2);
ylabel('||s||_2'); xlabel('iter (k)');
