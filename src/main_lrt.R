library(clrdag)
library(reticulate)
source_python('sim_gen.py')
np <- import("numpy")

LRT_test = function(d,N,T,delta,prob){
    d = as.integer(d)
    N = as.integer(N)
    T = as.integer(T)
    pval_mat = matrix(0,100,200)
    for (seed in 1:200){
        result = sim_gen(as.integer(seed),d,prob,N,T,delta)
        xs = result[[1]]
        W = result[[2]]
        edges = get_edges(d,W)
        edge_null_pair = edges[[1]]
        edge_alter_pair = edges[[2]]

        pval_ls = c()
        for (i in 1:100){
            edge1 = as.integer(edge_null_pair[[i]][[1]] + 1)
            edge2 = as.integer(edge_null_pair[[i]][[2]] + 1)
            D <- matrix(0, d, d)
            D[edge1,edge2] = 1
            out <- MLEdag(X=xs,D=D,tau=0.3,mu=1,rho=1.2,trace_obj=FALSE)
            if (is.null(out$pval)){
                pval = 0
            } else{
                pval <- out$pval
            }
            pval_ls[i] = pval
            }
        pval_mat[,seed] = pval_ls
        np$save(paste0("../data/LRT_nonlinear_",d,"_null_N_",N,"_T_",T,"_delta_",delta,".npy"), pval_mat)
        print(seed)
        }
}

LRT_test(50,10,100,1,0.1)
LRT_test(50,20,100,1,0.1)
LRT_test(50,40,100,1,0.1)
LRT_test(50,20,50,1,0.1)
LRT_test(50,20,200,1,0.1)
LRT_test(50,20,100,0.5,0.1)
LRT_test(50,20,100,2,0.1)
LRT_test(100,20,100,1,0.04)
LRT_test(150,20,100,1,0.02)