from inference import *
from nonlinear_learning import *

def index_to_graph(num,d):
    i = int(num/d)
    j = num%d
    return i,j

def sim_gen(seed,d,prob,N,T,delta):     
    np.random.seed(8)
    W0 = generate_W(d=d, prob=prob)
    
    np.random.seed(100+seed)
    K = 20
    c = 2
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
    
    return xs, W
    
def get_edges(d,W):
    np.random.seed(8)
    arr_index = np.arange(d**2).reshape([d,d])
    edge_null = arr_index[W==0]
    edge_alter = arr_index[W!=0]
    edge_null_select = np.random.choice(edge_null,100,replace=False)
    edge_alter_select = np.random.choice(edge_alter,100,replace=False)
    edge_null_pair = [index_to_graph(item,d) for item in edge_null_select]
    edge_alter_pair = [index_to_graph(item,d) for item in edge_alter_select]
    
    return edge_null_pair,edge_alter_pair