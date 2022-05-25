from inference import *
from nonlinear_learning import *

def index_to_graph(num,d):
    i = int(num/d)
    j = num%d
    return i,j

def cal_pvalue(seed,j,k,W0,b_ls,args):     
    np.random.seed(100+seed)
    K = args.K
    d = args.d
    N = args.N
    T = args.T
    c = args.c
    M = args.M
    B = args.B
    delta = args.delta
    h_size = args.h_size
    n_iter = args.n_iter
    v_dims = args.v_dims
    h_dims = args.h_dims
    size=int(N*T)
    arparams = np.r_[1,-0.5]
    maparam = np.r_[1]
    begin=500
    
    if args.graph_type == "nonlinear":
        y_ls=[]
        for i in range(d):
            for n in range(N):
                if n==0:
                    y=arma_generate_sample(arparams, maparam, T + begin)
                    y=y[begin:(T+begin)]
                else:
                    q=arma_generate_sample(arparams, maparam, T + begin)
                    put=copy.deepcopy(q[begin:(T+begin)])
                    y=np.concatenate((y, put), axis=0)
            y_ls.append(y)        
        xs, W =generate_iteraction(d,W0,N,T,c,delta,y_ls)
    if args.graph_type == "linear":
        W = W0
        xs = np.zeros((size, d))
        for i in range(d):
            for n in range(N):
                if n==0:
                    y=arma_generate_sample(arparams, maparam, T + begin)
                    y=y[begin:(T+begin)]
                else:
                    q=arma_generate_sample(arparams, maparam, T + begin)
                    put=copy.deepcopy(q[begin:(T+begin)])
                    y=np.concatenate((y, put), axis=0)
            xs[:, i] = y + xs.dot(W[i, :])
    b_all=b_ls[seed]
    root_all=root_relationship_all(b_all)
    result=cal_infer_SG_DRT(j=j,k=k,root_all=root_all,K = K,b_all=b_all,xs=xs,d=d,M=M,B=B,N=N,T=T,n_iter=n_iter,h_size=h_size,v_dims=v_dims,h_dims=h_dims)
    result.append(W[j,k])
    print("null value: ",W[j,k])
    print("p value: ",result)
    return result


def cal_graph(random0,W0,graph_type,args):
    np.random.seed(100+random0)
    N=args.N
    T=args.T
    c=args.c
    delta=args.delta
    n_ratio=0
    n_generate=2
    max_iter = args.max_iter
    d = args.d
    
    if  d==50 and graph_type == "nonlinear":
        lambda0 = 0.025
        w_threshold = 0.15
        h_units = 3
    elif  d==100 and graph_type == "nonlinear":
        lambda0 = 0.025
        w_threshold = 0.15
        h_units = 3     
    elif  d==150 and graph_type == "nonlinear":
        lambda0 = 0.025
        w_threshold = 0.15
        h_units = 3 
    elif  d==50 and graph_type == "linear":
        lambda0 = 0.025
        w_threshold = 0.25
        h_units = 10 
        
    size=int(N*T)
    arparams = np.r_[1,-0.5]
    maparam = np.r_[1]
    begin=500
    
    if graph_type == "nonlinear":
        y_ls=[]
        for i in range(d):
            for n in range(N):
                if n==0:
                    y=arma_generate_sample(arparams, maparam, T + begin)
                    y=y[begin:(T+begin)]
                else:
                    q=arma_generate_sample(arparams, maparam, T + begin)
                    put=copy.deepcopy(q[begin:(T+begin)])
                    y=np.concatenate((y, put), axis=0)
            y_ls.append(y)
        value=generate_iteraction(d,W0,N,T,c,delta,y_ls)
        xs=value[0]
        W=value[1]
    elif graph_type == "linear":
        W = W0
        xs = np.zeros((size, d))
        for i in range(d):
            for n in range(N):
                if n==0:
                    y=arma_generate_sample(arparams, maparam, T + begin)
                    y=y[begin:(T+begin)]
                else:
                    q=arma_generate_sample(arparams, maparam, T + begin)
                    put=copy.deepcopy(q[begin:(T+begin)])
                    y=np.concatenate((y, put), axis=0)
            xs[:, i] = y + xs.dot(W[i, :])   
                   
    size=xs.shape[0]
    each_size=int(size/2)
    xs1=copy.deepcopy(xs[0:each_size,:])
    xs2=copy.deepcopy(xs[each_size:size,:])
    
    torch.manual_seed(0)
    
    model=NotearsMLP(dims=[d, h_units, 1], bias=True)
    W_est1 = notears_nonlinear(model, xs1, lambda1=lambda0, lambda2=lambda0, w_threshold=w_threshold,max_iter=max_iter)
    revise_times=1
    if ut.is_dag(W_est1)==False:
        revise=0
        while revise==0:
            W_est1 = notears_nonlinear(model, xs1, lambda1=lambda0+revise_times*lambda0/5, lambda2=lambda0+revise_times*lambda0/5, w_threshold=w_threshold,max_iter=max_iter)
            if ut.is_dag(W_est1)==True:
                revise=1
            revise_times=revise_times+1

    W_est2 = notears_nonlinear(model, xs2, lambda1=lambda0, lambda2=lambda0, w_threshold=w_threshold,max_iter=max_iter)
    revise_times=1
    if ut.is_dag(W_est2)==False:
        revise=0
        while revise==0:
            W_est2 = notears_nonlinear(model, xs2, lambda1=lambda0+revise_times*lambda0/5, lambda2=lambda0+revise_times*lambda0/5, w_threshold=w_threshold,max_iter=max_iter)
            if ut.is_dag(W_est2)==True:
                revise=1
            revise_times=revise_times+1

    b_all=[copy.deepcopy(W_est1.T),copy.deepcopy(W_est2.T)]
    return(b_all)

def cal_true_graph(args,W0):
    d = args.d
    N = args.N
    T = args.T
    c = args.c
    delta = args.delta
    size=int(N*T)
    arparams = np.r_[1,-0.5]
    maparam = np.r_[1]
    begin=500

    y_ls=[]
    for i in range(d):
        for n in range(N):
            if n==0:
                y=arma_generate_sample(arparams, maparam, T + begin)
                y=y[begin:(T+begin)]
            else:
                q=arma_generate_sample(arparams, maparam, T + begin)
                put=copy.deepcopy(q[begin:(T+begin)])
                y=np.concatenate((y, put), axis=0)
        y_ls.append(y)        
    xs, W =generate_iteraction(d,W0,N,T,c,delta,y_ls)

    return W

def struct_learn(W0,graph_type,args):
    b_ls=[cal_graph(random0,W0,graph_type,args) for random0 in range(200)]
    return b_ls