from infer_utils import *
from real_utils import *
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='sugar')
parser.add_argument('-seed', '--seed', type=int, default=8)
parser.add_argument('-d', '--d', type=int, default=50)
parser.add_argument('-K', '--K', type=int, default=20)
parser.add_argument('-N', '--N', type=int, default=20)
parser.add_argument('-T', '--T', type=int, default=100)
parser.add_argument('-c', '--c', type=float, default=2)
parser.add_argument('-M', '--M', type=int, default=100)
parser.add_argument('-B', '--B', type=int, default=1000)
parser.add_argument('-delta', '--delta', type=float, default=1)
parser.add_argument('-prob', '--prob', type=float, default=0.1)
parser.add_argument('-h_size', '--h_size', type=int, default=100)
parser.add_argument('-n_iter', '--n_iter', type=int, default=300)
parser.add_argument('-v_dims', '--v_dims', type=int, default=5)
parser.add_argument('-h_dims', '--h_dims', type=int, default=2000)
parser.add_argument('-max_iter', '--max_iter', type=int, default=20)
parser.add_argument('-graph_type', '--graph_type', type=str, default="nonlinear")
args = parser.parse_args()


def run_sim(d,graph_type,N,T,delta):
    args.d = d
    args.graph_type = graph_type
    args.N = N
    args.T = T
    args.delta = delta
    
    if args.d == 50:
        args.prob = 0.1
    elif args.d == 100:
        args.prob = 0.04
    elif args.d == 150:
        args.prob = 0.02       
       
    np.random.seed(8)
    W0 = generate_W(d=args.d, prob=args.prob)
    
    b_ls = struct_learn(W0,args.graph_type,args)
    
    if args.graph_type=="nonlinear":
        W = cal_true_graph(args,W0)
        args.h_dims = 2000
    elif args.graph_type=="linear":
        W = W0
        args.h_dims = 1000

    np.random.seed(8)
    arr_index = np.arange(args.d**2).reshape([args.d,args.d])
    edge_null = arr_index[W==0]
    edge_alter = arr_index[W!=0]
    edge_null_select = np.random.choice(edge_null,100,replace=False)
    edge_alter_select = np.random.choice(edge_alter,100,replace=False)
    edge_null_pair = [index_to_graph(item,args.d) for item in edge_null_select]
    edge_alter_pair = [index_to_graph(item,args.d) for item in edge_alter_select]
    
    result_all = []
    for edge in edge_null_pair:
        j = edge[0] 
        k = edge[1] 
        result_iter=[]
        for seed in tqdm(range(200)):
            result=cal_pvalue(seed,j,k,W0,b_ls,args)
            result_iter.append(result)  
            print("p value every random: ",result_iter)
        result_all.append(result_iter)
        path = "../data/"+args.graph_type+"_"+str(args.d)+"_null_N_"+str(args.N)+"_T_"+str(args.T)+"_delta_"+str(args.delta)
        np.save(path,result_all)
        
    result_all = []
    for edge in edge_alter_pair:
        j = edge[0] 
        k = edge[1] 
        result_iter=[]
        for seed in tqdm(range(200)):
            result=cal_pvalue(seed,j,k,W0,b_ls,args)
            result_iter.append(result)  
            print("p value every random: ",result_iter)
        result_all.append(result_iter)
        path = "../data/"+args.graph_type+"_"+str(args.d)+"_alter_N_"+str(args.N)+"_T_"+str(args.T)+"_delta_"+str(args.delta)
        np.save(path,result_all)

def main():
    run_sim(50,"nonlinear",10,100,1)
    run_sim(50,"nonlinear",20,100,1)
    run_sim(50,"nonlinear",40,100,1)
    run_sim(50,"nonlinear",20,50,1)
    run_sim(50,"nonlinear",20,200,1)
    run_sim(50,"nonlinear",20,100,0.5)
    run_sim(50,"nonlinear",20,100,2)
    
    run_sim(100,"nonlinear",20,100,1)
    run_sim(150,"nonlinear",20,100,1)
    
    run_real("low")
    run_real("high")
    
if __name__ == '__main__':
    main()