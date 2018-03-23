% Lhalf regularized logistic regression (not distributed)
clc
clear all;

% Generate problem data
rand('seed', 0);
randn('seed', 0);

n = 200;  % sample
p = 100;   % feature

beta_int = sprandn(p, 1, 0.1);  % N(0,1), 10% sparse
% index_beta_non0=find(beta_int~=0);
% beta_int(index_beta_non0)=beta_int(index_beta_non0)*5;
beta_zero = randn(1);            % random intercept

X = sprandn(n, p, 10/p);
Ytrue = sign(X*beta_int + beta_zero);
% noise is function of problem size use 0.1 for large problem
Y = sign(X*beta_int + beta_zero + sqrt(0.1)*randn(n,1)); % labels with noise

A = spdiags(Y, 0, n, n) * X;

ratio = sum(Y == 1)/(n);
mu = 0.1 * 1/n * norm((1-ratio)*sum(A(Y==1,:),1) + ratio*sum(A(Y==-1,:),1), 'inf');

beta_true = [beta_zero; beta_int];

% Solve problem

[beta history] = lhalf_logreg(A, Y, mu, 1.0, 1.0);

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
