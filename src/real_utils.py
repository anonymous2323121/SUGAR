from inference import *
from nonlinear_learning import *
import networkx as nx

def calculate_interaction_graph_real(xs,max_iter,d,lambda0,w_threshold):
    judge1=0
    judge2=0
    N=int(xs.shape[0]/316)
    T=316
    size=xs.shape[0]
    each_size=int(size/2)
    xs1=copy.deepcopy(xs[0:each_size,:])
    xs2=copy.deepcopy(xs[each_size:size,:])
    model=NotearsMLP(dims=[d, 3, 1], bias=True)
    W_est1 = notears_nonlinear(model, xs1, lambda1=lambda0, lambda2=lambda0, w_threshold=w_threshold,max_iter=max_iter)
    revise_times=1
    if ut.is_dag(W_est1)==False:
        judge1=1
        revise=0
        while revise==0:
            print("revise: ",revise_times)
            W_est1 = notears_nonlinear(model, xs1, lambda1=lambda0+revise_times*lambda0/5, lambda2=lambda0+revise_times*lambda0/5, w_threshold=w_threshold,max_iter=max_iter)
            if ut.is_dag(W_est1)==True:
                revise=1
            revise_times=revise_times+1
    W_est2 = notears_nonlinear(model, xs2, lambda1=lambda0, lambda2=lambda0, w_threshold=w_threshold,max_iter=max_iter)
    revise_times=1
    if ut.is_dag(W_est2)==False:
        judge2=1
        revise=0
        while revise==0:
            print("revise: ",revise_times)
            W_est2 = notears_nonlinear(model, xs2, lambda1=lambda0+revise_times*lambda0/5, lambda2=lambda0+revise_times*lambda0/5, w_threshold=w_threshold,max_iter=max_iter)
            if ut.is_dag(W_est2)==True:
                revise=1
            revise_times=revise_times+1
    b_all=[copy.deepcopy(W_est1.T),copy.deepcopy(W_est2.T),judge1,judge2]
    return(b_all)

def cal_pvalue_real(j,k,K,b_ls,xs):     
    T=316
    N=int(xs.shape[0]/316)
    b_all=b_ls
    ##### root1
    G=nx.DiGraph()
    G.add_nodes_from(range(d))
    for m in range(d):
        for n in range(d):
            if b_all[0][m,n]!=0:
                G.add_edge(n,m)
    root1=[]
    for m in range(d):
        root1.append(list(nx.ancestors(G, m)))
    ##### root2
    G=nx.DiGraph()
    G.add_nodes_from(range(d))
    for m in range(d):
        for n in range(d):
            if b_all[1][m,n]!=0:
                G.add_edge(n,m)
    root2=[]
    for m in range(d):
        root2.append(list(nx.ancestors(G, m)))
    ###### root_all    
    root_all=[root1,root2]
   
    result = cal_infer_SG_DRT(j=j,k=k,root_all=root_all,K = K,b_all=b_all,xs=xs,d=d,M=100,B=1000,N=N,T=T,n_iter=300,h_size=100,v_dims=5,h_dims=2000)
    return result

def run_real(data_type):
    if data_type == "low":
        HCP_test=np.load("../data/HCP_low.npy",allow_pickle=True)
    elif data_type == "high":
        HCP_test=np.load("../data/HCP_high.npy",allow_pickle=True)
    xs=HCP_test
    xs=(xs-np.mean(xs))/np.std(xs)
    d=xs.shape[1]
    
    max_iter=20
    lambda0=0.025
    w_threshold=0.3
    b_ls = calculate_interaction_graph_real(xs,max_iter,d,lambda0,w_threshold) 
    
    K = 20
    result_all = []
    for j in range(d):
        result_iter=[]
        for k in range(d):           
            result=cal_pvalue_real(j,k,K,b_ls,xs)
            result_iter.append(result)
        result_all.append(result_iter)
        if data_type == "low":
            np.save("../data/HCP_low_pvalue.npy",result_all)
        elif data_type == "high":
            np.save("../data/HCP_high_pvalue.npy",result_all)