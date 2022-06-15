from inference import *
from nonlinear_learning import *
import networkx as nx
import numpy as np
import copy
from random import randint

def calculate_interaction_graph_real(xs,max_iter,d,lambda0,w_threshold):
    torch.manual_seed(0)
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
            
def print_real(data_type):
    if data_type == "low":
        result_all = np.load("../data/HCP_low_pvalue.npy",allow_pickle=True)
    elif data_type == "high":
        result_all = np.load("../data/HCP_high_pvalue.npy",allow_pickle=True)
        
    pvalue_ls=[]
    for m in range(len(result_all)):
        for k in range(len(result_all[m])):
            pvalue_ls.append(result_all[m][k][0])

    order=np.argsort(pvalue_ls)

    pvalue_order=[]
    for m in range(len(pvalue_ls)):
        pvalue_order.append(pvalue_ls[order[m]])

    M=len(pvalue_order)

    R=0
    for m in range(M):
        R=R+1/(1+m)

    q=0.05
    for m in range(M):
        k=M-m-1
        if pvalue_order[k]<= k*q/(M*R):
            M0=k
            break

    W=np.zeros([127,127])
    count=0
    for m in range(len(result_all)):
        for k in range(len(result_all[m])):
            if pvalue_ls[count]<=pvalue_order[M0]:
                W[m,k]=1
            count=count+1
            
    d=W.shape[0]
    G=nx.DiGraph()
    G.add_nodes_from(range(d))
    for m in range(d):
        for n in range(d):
            if W[m,n]!=0:
                G.add_edge(n,m)
    module_name=np.load("../data/module_name.npy",allow_pickle=True)
    color_map = []
    node1=[]
    node2=[]
    node3=[]
    node4=[]
    for node in G:
        if module_name[node]=="Auditory":
            color_map.append('blue') 
            node1.append(node)
        if module_name[node]=="Default mode":
            color_map.append('red')    
            node2.append(node)
        if module_name[node]=="Visual":
            color_map.append('green')   
            node3.append(node)
        if module_name[node]=="Fronto-parietal Task Control":
            color_map.append('yellow')  
            node4.append(node)
    relationship=np.zeros([4,4])
    
    def return_position(m):
        if m in node1:
            value=0
        elif m in node2:
            value=1
        elif m in node3:
            value=2
        elif m in node4:
            value=3
        return int(value)
    for i in range(127):
        for j in range(127):
            if W[i,j]!=0:
                relationship[return_position(i),return_position(j)]=relationship[return_position(i),return_position(j)]+1
                
    show=relationship+relationship.T
    for m in range(4):
        show[m,m]=show[m,m]/2
    print(show)