size(connectivity)

 K = 10
 nreps = 10
 gamma_min_ratio = 0.05
 maxit = 100
 fullit = 100
 tol=1e-4
 ngam= 10

    
Age = double(response.Ab42by40);
connectivity = double(connectivity);

ngam= 10
nfold = 5 
indices = crossvalind('Kfold',Age,nfold);

mse = 0
mses = []
for i = 1:nfold
    test = (indices == i); 
    train = ~test;
    %[alpha_final,Lambda_final,Beta_final, coefM_final,coefM_final_vec,min_MSE_test,gamma_seq,MSE_set] = SBL_tuning_gamma(connectivity(:,:,train),Age(train),connectivity(:,:,test),Age(test),K,nreps,ngam, gamma_min_ratio,maxit,fullit,tol);
    pred = mean(Age(train))
    mse_temp = sqrt( mean((pred-Age(test)).^2)  )
    mse = mse + mse_temp
    mses(i) = mse_temp
end
mse = mse/ nfold
std(mses)

