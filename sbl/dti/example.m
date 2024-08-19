 
size(connectivity)

 K = 10
 nreps = 10
 gamma_min_ratio = 0.05
 maxit = 1000
 fullit = 100
 tol=1e-5
 ngam= 10

Age = double(response)
connectivity = double(connectivity)

ngam= 10
nfold = 5 
indices = crossvalind('Kfold',Age,nfold);

mse = 0
mses = []
for i = 1:nfold
    test = (indices == i); 
    train = ~test;
    [alpha_final,Lambda_final,Beta_final, coefM_final,coefM_final_vec,min_MSE_test,gamma_seq,MSE_set] = SBL_tuning_gamma(connectivity(:,:,train),Age(train),connectivity(:,:,test),Age(test),K,nreps,ngam, gamma_min_ratio,maxit,fullit,tol);
    mse = mse + sqrt(min_MSE_test)
    mses(i) = sqrt(min_MSE_test)
end
mse = mse/ nfold
std(mses)
dlmwrite('LTPR_dti.csv',mses )



[alpha_final,Lambda_final,Beta_final, coefM_final,coefM_final_vec,min_MSE_test,gamma_seq,MSE_set]   = SBL_tuning_gamma(connectivity,Age,connectivity,Age,K,10,nreps,gamma_min_ratio,maxit,fullit,tol)
save vector.mat coefM_final_vec
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%
% Output:
%   alpha_final: scalar, estimate of alpha under the optimal gamma
%   Lambda_final: Kx1 vector, estimate of Lambda under the optimal gamma
%   Beta_final: VxK matrix, estimate of Beta under the optimal gamma
%   coefM_final: VxV matrix, Beta_final * diag(Lambda_final) * Beta_final'
%   coefM_final_vec: 2 x upper triangular of coefM_final 
%   min_MSE_test: minimum MSE of test set
%   gamma_seq: gamma sequence for tuning
%   MSE_set: set of MSEs on test set across gamma values
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%