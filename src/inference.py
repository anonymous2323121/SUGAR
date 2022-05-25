import numpy as np
import random
import copy
import scipy
from random import sample
import matplotlib.pyplot as plt
from synthetic import *
import tensorflow as tf
import utils_tools 
import statsmodels.api as sm
from statsmodels.tsa.arima_process import arma_generate_sample
from sklearn import neural_network
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import cross_val_predict


def root_relationship(b_):
    '''
    calculate the ancestors for each node
    input:
        graph matrix
    return:
        list of ancestors for each node
    '''
    root=[]
    d=int(b_.shape[0])
    for m in range(d):
        if(np.sum(b_[m,:]!=0)==0):
            root.append([])
        else:
            basis=[]
            extra=[]
            new=[]
            for k in range(d):
                if b_[m,k]!=0:
                    new.append(k)
            basis=copy.deepcopy(new)
            stop=1
            while stop==1:
                new=[]
                for k in range(len(basis)):
                    root_m=basis[k]
                    extra.append(root_m)
                    if (root_m < m):
                        for u in range(len(root[root_m])):
                            extra.append(root[root_m][u])
                    else:
                        for u in range(d):
                            if b_[root_m,u]!=0:
                                new.append(u)
                basis=copy.deepcopy(new)
                if basis==[]:
                    stop=0
            extra=sorted(set(extra))
            root.append(extra)
    return root

def root_relationship_all(b_all):
    '''
    calculate the ancestors for the graph matrix of each random split
    '''
    root_all=[]
    for l in range(2):
        root_all.append(root_relationship(b_all[l]))
    return(root_all)    

def cal_ancestor(root,j,k):
    '''
    calculate the ancestors (without k) for each node j given the root object
    '''
    act_j=copy.deepcopy(root[j])
    try:
        act_j.remove(k)
    except ValueError:
        pass
    return(act_j)

def generate_f(x,choose_seed):
    '''
    data generating function with equal probability to be sine or cosine.
    '''
    if choose_seed[0]<1:
        value=np.sin(x)
    else:
        value=np.cos(x)
    return value

def generate_iteraction(d,W,N,T,c,delta,y_ls):
    '''
    data and graph matrix generating function
    input:
        dimension of the graph (d), original graph (W), number of subjects (N), time length of each series (T), 
        paramethers to control edge strength (c and delta), original AR process before transformation (y_ls)
    return:
        generated series (xs) and graph matrix (W_new)
    '''
    # we fix a seed for the generating process so that the interaction setting and graph matrix will be the same
    # across the repetitions. The only randomness will come from the original AR process (y_ls).
    random.seed(8)
    np.random.seed(8)
    W_new=np.zeros([d,d])
    size=int(N*T)
    arparams = np.r_[1,-0.5]
    maparam = np.r_[1]
    xs = np.zeros((size, d))
    for i in range(d):
        y=y_ls[i]
        parents= (W[i,:]!=0)
        # total number of parents for node i
        n_parents=int(np.sum(W[i,:]!=0))
        parents_position=[]
        for m in range(d):
            if W[i,m]!=0:
                parents_position.append(m)
        if n_parents==0:
            xs[:,i]=y
        else:
            xs[:,i]=y
            xs_parents=xs[:,parents]
            # set the sparsity of parent edges
            n_U=int(n_parents*(n_parents+1)/2+n_parents)
            U=np.random.uniform(0.5,1.5,n_U)*delta
            select=np.random.uniform(-1,1,n_U)
            U[select<0]=-U[select<0]
            n_select=int(c*n_U/n_parents)
            if n_select < n_U:
                select_sample=sample(range(n_U),int(n_U-n_select))
                U[select_sample]=0
            count=0
            for k in range(n_parents):
                for m in range(int(n_parents-k)):
                    # generate interaction terms by using sine or cosine function with equal probability
                    choose=int(m+k)
                    choose_seed=sample(range(2),1)
                    part1=generate_f(xs_parents[:,k],choose_seed)
                    choose_seed=sample(range(2),1)
                    part2=generate_f(xs_parents[:,choose],choose_seed)
                    xs[:,i]= xs[:,i]+U[count]*part1*part2
                    if(U[count]!=0):
                        W_new[i,parents_position[k]]=1
                        W_new[i,parents_position[choose]]=1
                    count=count+1
            for k in range(n_parents):
                    # generate linear terms by using sine or cosine function with equal probability
                    choose_seed=sample(range(2),1)
                    part=generate_f(xs_parents[:,k],choose_seed)
                    xs[:,i]= xs[:,i]+U[count]*part
                    if(U[count]!=0):
                        W_new[i,parents_position[k]]=1
                    count=count+1  
    return xs, W_new  

def split_data(xs,l):
    # split the subjects into two groups
    size=xs.shape[0]
    each_size=int(size/2)
    xs1=copy.deepcopy(xs[0:each_size,:])
    xs2=copy.deepcopy(xs[each_size:size,:])    

    # set one group as training (estimate learners) and another as testing (evaluate learners)
    if l==0:
        train=copy.deepcopy(xs1)
        test=copy.deepcopy(xs2)
    else:
        train=copy.deepcopy(xs2)
        test=copy.deepcopy(xs1)    
        
    return train,test,each_size 

def cal_diff2(GAN_result,xk_train,B,each_size,M):
    # generate random u,v
    u=np.random.normal(0,1,B)
    u_ls=[]
    
    # construct diff2
    part1_cos=[]
    part1_sin=[]
    part2_cos=[]
    part2_sin=[]
    extract=[]
    extract=np.zeros([each_size,M])
    for m in range(each_size):
        q=GAN_result[0][0][m].numpy()
        q=q.reshape(q.shape[0],)
        extract[m,:]=q
        
    for b1 in range(B):
        part1_first=u[b1]*xk_train
        part1_cos.append(np.cos(part1_first))
        part1_sin.append(np.sin(part1_first))
        record=u[b1]*extract
        q1=np.sum(np.cos(record),axis=1)/M
        q1=q1.reshape(q1.shape[0],1)
        part2_cos.append(q1)
        q2=np.sum(np.sin(record),axis=1)/M
        q2=q2.reshape(q2.shape[0],1)
        part2_sin.append(q2)
        
    diff2=[]
    for b1 in range(B):
        part1=part1_cos[b1]
        part2=part2_cos[b1]
        diff2.append(part1-part2)
        u_ls.append(u[b1])
        part1=part1_sin[b1]
        part2=part2_sin[b1]
        diff2.append(part1-part2) 
        u_ls.append(u[b1])
    
    return diff2, u_ls

def cal_diff2_mean(xk_train,B,each_size,M):
    # generate random u,v
    u=np.random.normal(0,1,B)
    u_ls=[]
    
    # construct diff2
    part1_cos=[]
    part1_sin=[]
    part2_cos=[]
    part2_sin=[]
    for b1 in range(B):
        part1_first=u[b1]*xk_train
        part1_cos.append(np.cos(part1_first))
        part1_sin.append(np.sin(part1_first))
        part2_cos.append(np.cos(part1_first))
        part2_sin.append(np.sin(part1_first))
        
    diff2=[]
    for b1 in range(B):
            part1=part1_cos[b1]
            part2=np.mean(part2_cos[b1])
            diff2.append(part1-part2)
            u_ls.append(u[b1])
            part1=part1_sin[b1]
            part2=np.mean(part2_sin[b1])
            diff2.append(part1-part2) 
            u_ls.append(u[b1])
    
    return diff2, u_ls

def cal_position(diff1,diff2,B,N,T,each_size,K):
    # calculate I_bt
    I_bt=[]
    for b in range(int(2*B)):
        I_bt.append(diff1*(diff2[b]))

    # calculate I_b
    I_b=[]
    for b in range(int(2*B)):
        I_b.append(2*np.sum(I_bt[b])/(N*T))
        
    # calculate I_b0
    divide = int(each_size/ K)
    I_bt0 = []
    for m in range(2 * B):
        I_bt0.append((I_bt[m] - I_b[m]) / np.sqrt(K))
        
    store = np.zeros([divide, 2 * B])
    store = np.asmatrix(store)
    for m in range(0, divide):
        choose = np.array(list(range(0, K))) + m * K
        for mm in range(0, 2 * B):
            store[m, mm] = np.sum(I_bt0[mm][choose])
    # select b        
    sigma=store.T*store
    sigma = sigma / divide
    diag = [sigma[qq,qq] for qq in range(len(sigma))]
    diag=np.sqrt(diag)
    b_value=np.abs(I_b)/diag
    b_position=np.argmax(b_value)
    
    return b_position


def cal_pvalue_SG(GAN_result,u_ls,xp2,xj_test,xk_test,b_position,each_size,M,N,T,K):
    uu=u_ls[b_position]
    diff1=xj_test-xp2
    diff1=diff1.reshape(diff1.shape[0],1)
    extract=[]
    for m in range(each_size):
        q=GAN_result[1][0][m].numpy()
        extract.append(q)
    part1_first=uu*xk_test
    part1_cos=np.cos(part1_first)
    part1_sin=np.sin(part1_first)
    record=[]
    for m in range(each_size):
        record.append(np.cos(uu*(extract[m])))
    part2_cos=np.sum(record,axis=1)/M
    record=[]
    for m in range(each_size):
        record.append(np.sin(uu*(extract[m])))
    part2_sin=np.sum(record,axis=1)/M
    if (b_position%2==0):
        part1=part1_cos
        part2=part2_cos
        diff2=part1-part2
    else:
        part1=part1_sin
        part2=part2_sin
        diff2=part1-part2
    I_bt=diff1*(diff2)
    I_b=2*np.sum(I_bt)/(N*T)
    divide = int(each_size/ K)
    I_bt0=(I_bt - I_b) / np.sqrt(K)
    res=0
    for m in range(0, divide):
        choose = np.array(list(range(0, K))) + m * K
        ww=np.sum(I_bt0[choose])
        res = res + ww*ww
    statistics=np.sqrt(each_size)*np.abs(I_b)/np.sqrt(res/divide)
    pvalue_SG = 2*scipy.stats.norm(0, 1).cdf(-statistics)
    
    return pvalue_SG

def cal_pvalue_SG_mean(u_ls,xp2,xj_test,xk_test,b_position,each_size,M,N,T,K):
    uu=u_ls[b_position]
    diff1=xj_test-xp2
    diff1=diff1.reshape(diff1.shape[0],1)
    part1_first=uu*xk_test
    part1_cos=np.cos(part1_first)
    part1_sin=np.sin(part1_first)
    part2_cos=np.cos(part1_first)
    part2_sin=np.sin(part1_first)
    if (b_position%2==0):
        part1=part1_cos
        part2=np.mean(part2_cos)
        diff2=part1-part2
    else:
        part1=part1_sin
        part2=np.mean(part2_sin)
        diff2=part1-part2
    I_bt=diff1*(diff2)
    I_b=2*np.sum(I_bt)/(N*T)
    divide = int(each_size/ K)
    I_bt0=(I_bt - I_b) / np.sqrt(K)
    res=0
    for m in range(0, divide):
        choose = np.array(list(range(0, K))) + m * K
        ww=np.sum(I_bt0[choose])
        res = res + ww*ww
    statistics=np.sqrt(each_size)*np.abs(I_b)/np.sqrt(res/divide)
    pvalue_SG = 2*scipy.stats.norm(0, 1).cdf(-statistics)

    return pvalue_SG

def cal_pvalue_DRT(diff1,diff2,N,T,each_size,K):
    I_bt=diff1*(diff2)
    I_b=2*np.sum(I_bt)/(N*T)
    divide = int(each_size/ K)
    I_bt0=(I_bt - I_b) / np.sqrt(K)
    res=0
    for m in range(0, divide):
        choose = np.array(list(range(0, K))) + m * K
        ww=np.sum(I_bt0[choose])
        res = res + ww*ww
    statistics=np.sqrt(each_size)*np.abs(I_b)/np.sqrt(res/divide)
    pvalue_DRT = 2*scipy.stats.norm(0, 1).cdf(-statistics)
    return pvalue_DRT
    
def cal_infer_SG_DRT(j,k,root_all,K,b_all,xs,d,M,B,N,T,n_iter,h_size,v_dims,h_dims):
    '''
    testing the edge with both Sugar (SG) and Double Regression Testing (DRT)
    input:
        node j,k (j,k), ancestors information (root_all), choices of hyperparameter K (K), 
        estimated graph of each split (b_all), data series (xs), dimension of the graph (d), 
        number of generated samples (M), number of subjects (N), length of each series (T), 
        number of iterations for training (n_iter), number of hidden units (h_size)
    return:
        the list: [pvalue of SG, pvalue of DRT, [j,k]]
    '''

    # set random seed
    np.random.seed(8)
    random.seed(8)
    tf.random.set_seed(8)
    
    pvalue_SG = []
    pvalue_DRT = []
    for l in range(2):
        '''
        SUGAR Test
        '''
        # split the training and testing set
        train, test, each_size = split_data(xs,l)
        
        # calculate the ancestors
        act_j=cal_ancestor(root_all[l],j,k)
        
        # variables for training
        z_train=train[:,act_j]
        z_test=test[:,act_j]
        xj_train=train[:,j]
        xj_test=test[:,j]
        xk_train=train[:,[k]]
        xk_test=test[:,[k]]
        dim_z=z_train.shape[1]

        if (k in root_all[l][j]) == False:
            '''
            k doesn't belong to ancestors of j so the p-value is set as 1
            '''
            
            '''
            SUGAR Test 
            '''
            pvalue_SG.append(1)
            
            '''
            Double Rregression Test 
            '''
            pvalue_DRT.append(1)
            
        elif (cal_ancestor(root_all[l],j,k)==[]) and (root_all[l][j]!=[]):
            '''
            when k is the only ancestor of j 
            '''  
            xp1=np.mean(xj_train)
            xp2=np.mean(xj_test)
            
            # calculate the first part of the test statistics
            diff1=xj_train-xp1
            diff1=diff1.reshape(diff1.shape[0],1)
            
            # calculate the second part of the test statistics
            diff2, u_ls = cal_diff2_mean(xk_train,B,each_size,M)

            # get the position
            b_position = cal_position(diff1,diff2,B,N,T,each_size,K)
            
            # calculate pvalue for Sugar
            pvalue_SG.append(cal_pvalue_SG_mean(u_ls,xp2,xj_test,xk_test,b_position,each_size,M,N,T,K))
            
            # update the first part of the test statistics for another split
            diff1=xj_test-xp2
            diff1=diff1.reshape(diff1.shape[0],1)
            
            # update the second part of the test statistics for another split
            xk_train=xk_train.reshape(xk_train.shape[0],)
            xk_test=xk_test.reshape(xk_test.shape[0],)
            xp2=np.mean(xk_test)
            diff2=xk_test-xp2
            diff2=diff2.reshape(diff2.shape[0],1)
            
            pvalue_DRT.append(cal_pvalue_DRT(diff1,diff2,N,T,each_size,K))   
        else:
            # supervised learning algorithm
            nn_unit = neural_network.MLPRegressor(hidden_layer_sizes=(h_size,),activation='relu', solver='adam', max_iter=2000)
            regressormodel = nn_unit.fit(z_train, xj_train)
            xp1 = nn_unit.predict(z_train)
            xp2 = nn_unit.predict(z_test)       
        
            # CGAN
            GAN_result=utils_tools.gcit_tools(x_train=xk_train,z_train=z_train,x_test=xk_test,z_test=z_test,v_dims=v_dims,h_dims=h_dims,M = M, batch_size=64, n_iter=n_iter, standardise =False,normalize=True)        

            # calculate the first part of the test statistics
            diff1=xj_train-xp1
            diff1=diff1.reshape(diff1.shape[0],1)

            # calculate the second part of the test statistics
            diff2, u_ls = cal_diff2(GAN_result,xk_train,B,each_size,M)

            # get the position
            b_position = cal_position(diff1,diff2,B,N,T,each_size,K)

            # calculate pvalue for Sugar
            pvalue_SG.append(cal_pvalue_SG(GAN_result,u_ls,xp2,xj_test,xk_test,b_position,each_size,M,N,T,K))

            # update the first part of the test statistics for another split
            diff1=xj_test-xp2
            diff1=diff1.reshape(diff1.shape[0],1)

            # supervised learning algorithm
            nn_unit = neural_network.MLPRegressor(hidden_layer_sizes=(h_size,),activation='relu', solver='adam', max_iter=2000)
            xk_train=xk_train.reshape(xk_train.shape[0],)
            xk_test=xk_test.reshape(xk_test.shape[0],)
            regressormodel = nn_unit.fit(z_train, xk_train)
            xp2 = nn_unit.predict(z_test)

            # update the second part of the test statistics for another split
            diff2=xk_test-xp2
            diff2=diff2.reshape(diff2.shape[0],1)

            pvalue_DRT.append(cal_pvalue_DRT(diff1,diff2,N,T,each_size,K))
        
    # save 4 digits of the p value
    result_ls=[]
    result_ls.append(round(2*np.minimum(pvalue_SG[0],pvalue_SG[1]),4))
    result_ls.append(round(2*np.minimum(pvalue_DRT[0],pvalue_DRT[1]),4))
    result_ls.append([j,k])
    return(result_ls)           
        