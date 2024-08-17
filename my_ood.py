import os
import time
from util.args_loader import get_args
import torch
import faiss
import numpy as np
from glob import glob
from my_faiss_index import LocalFaissIndex
import pandas as pd
import math
from scipy import stats

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_args()

os.environ["CUDA_VISIBLEgt_pose_x_DEVICES"] = args.gpu

def get_pose_diff_from_csv(train_row, test_pose_x, test_pose_q):
    train_pose_x = train_row['gt_pose_x']
    train_pose_q = train_row['gt_pose_q']
    train_pose_x = torch.from_numpy(np.array([float(x.strip(' []')) for x in train_pose_x.split(',')]))
    train_pose_q = torch.from_numpy(np.array([float(x.strip(' []')) for x in train_pose_q.split(',')]))
    error_x = torch.linalg.norm(torch.Tensor(test_pose_x-train_pose_x)).numpy()
    d = torch.abs(torch.sum(torch.matmul(test_pose_q,train_pose_q))) 
    d = torch.clamp(d, -1., 1.) # acos can only input [-1~1]
    theta = (2 * torch.acos(d) * 180/math.pi).numpy()    
    return error_x, theta

def load_features(files_path, is_normalize):    
    files_npy = glob(files_path + '/*.npy')
    feats = []
    names = []
    for f in files_npy:
        feat = np.load(f)
        #feat = np.expand_dims(feat, axis=0)
        filename = os.path.splitext(os.path.basename(f))[0]
        feats.append(feat)
        names.append(filename)        
    feats = np.array(feats)
    if is_normalize:
        normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)    
        prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))
        feats = prepos_feat(feats)
    return feats, names
    
def find_ood_by_similarity(train_path, test_path, ood_list, thr_sim):
    #load train features to faiss
    ood = []
    for o in ood_list:
        ood.append(os.path.splitext(os.path.basename(o.replace("\"","")))[0])
    is_normalize = False
    ftrain, train_names = load_features(train_path, is_normalize)
    ftest, test_names = load_features(test_path, is_normalize)
    if is_normalize:
        train_faiss = faiss.IndexFlatL2(ftrain.shape[1])
        train_faiss.add(ftrain)
    else:    
        train_faiss = LocalFaissIndex()
        #insert all train vector to faiss
        for i in range(len(ftrain)):
            feat = ftrain[i]
            filename = train_names[i]
            feat = np.expand_dims(feat, axis=0)
            idx, vector = train_faiss.insert_to_index(feat, filename)           
        train_faiss.save('train_faiss', 'train_faiss_meta')    
    #go over test images and find best match in train
    if is_normalize:
        thrs = [0.3, 0.4, 0.5]
    else:
        thrs = [0.7, 0.8, 0.9]
    for thr_sim in thrs:
        for k in (40, 60, 80):
            ood_by_sim = []
            for i in range(len(ftest)):
                feat = ftest[i]
                filename = test_names[i]                
                feat = np.expand_dims(feat, axis=0)
                if is_normalize:                    
                    scores, _ = train_faiss.search(feat, k)
                    score = -scores[:,-1]                
                    if score > -thr_sim:
                        ood_by_sim.append(filename)           
                else:
                    idx, scores, names = train_faiss.search(feat, 200)       
                    score = scores[k]                                
                    if score < thr_sim:
                        ood_by_sim.append(filename)           
            correct = 0
            not_correct = 0                   
            for o in ood_by_sim:
                if o in ood:
                    correct += 1
                else:
                    not_correct += 1
            correct_ratio = 0
            if len(ood_by_sim) > 0:
                correct_ratio = correct/(correct+not_correct)                
            print('thr_sim: ' +str(thr_sim) + ' k: ' +str(k) + ' accuracy: ' + str(correct_ratio))
       
def find_ood_by_distance(train_path, test_path, th_x, th_q):   
    train_df = pd.read_csv(train_path+'/dfnet_res.csv')
    test_df = pd.read_csv(test_path+'/dfnet_res.csv')
    train_df = train_df.reset_index()  # make sure indexes pair with number of rows
    test_df = test_df.reset_index()  # make sure indexes pair with number of rows
    ood_list = []
    #loop over all poses and find images where test pose if far from all train poses
    for i1, test_row in test_df.iterrows():
        test_pose_x = test_row['gt_pose_x']
        test_pose_q = test_row['gt_pose_q']
        test_pose_x = torch.from_numpy(np.array([float(x.strip(' []')) for x in test_pose_x.split(',')]))
        test_pose_q = torch.from_numpy(np.array([float(x.strip(' []')) for x in test_pose_q.split(',')]))
        cnt_far_neighbors = 0
        for i2, train_row in train_df.iterrows():
            error_x, theta = get_pose_diff_from_csv(train_row,test_pose_x, test_pose_q)
            if error_x > th_x or theta > th_q:                
                cnt_far_neighbors += 1
        if cnt_far_neighbors == train_df.shape[0]:
            ood_list.append(test_row['filename'])
            
    print(len(ood_list))
    return ood_list           

def find_nn_by_distance(train_path, test_path, test_image):      
    train_df = pd.read_csv(train_path+'/dfnet_res.csv')
    test_df = pd.read_csv(test_path+'/dfnet_res.csv')
    train_df = train_df.reset_index()  # make sure indexes pair with number of rows
    test_df = test_df.reset_index()  # make sure indexes pair with number of rows
    nn_list_x = {}
    nn_list_theta = {}
    min_error_x = 100
    min_theta = 100
    #loop over all poses and find images where test pose if far from all train poses
    for i1, test_row in test_df.iterrows():
        if test_image not in test_row['filename']:
            continue
        #test_row = test_df[test_image]        
        test_pose_x = test_row['gt_pose_x']
        test_pose_q = test_row['gt_pose_q']
        test_pose_x = torch.from_numpy(np.array([float(x.strip(' []')) for x in test_pose_x.split(',')]))
        test_pose_q = torch.from_numpy(np.array([float(x.strip(' []')) for x in test_pose_q.split(',')]))
        for i2, train_row in train_df.iterrows():
            error_x, theta = get_pose_diff_from_csv(train_row,test_pose_x, test_pose_q)      
            train_filename = train_row['filename']
            if error_x < min_error_x:
                min_error_x = error_x
                nn_list_x[train_filename] = min_error_x
            if theta < min_theta:
                min_theta = theta
                nn_list_theta[train_filename] = min_theta
                      
        
    nn_list_x = dict(sorted(nn_list_x.items(), key=lambda item: item[1]))
    nn_list_theta = dict(sorted(nn_list_theta.items(), key=lambda item: item[1]))
    return nn_list_x, nn_list_theta


def get_ood_list(train_path, test_path, th_x, th_q, is_find_ood):
    file_path = 'ood_list_'+str(th_x)+'x_'+str(th_q)+'q.txt'

    if is_find_ood:
        ood_list = find_ood_by_distance(train_path, test_path, th_x, th_q)     
        with open(file_path, 'w') as file:
            # Join the list elements into a single string with a newline character
            data_to_write = '\n'.join(ood_list)        
            # Write the data to the file
            file.write(data_to_write)
        ood = ood_list
    else:
        f = open(file_path, 'r')
        ood_list = f.read()
        f.close()
        ood = ood_list.replace('\n','').replace('(','').replace(')','').split(',')    
    
    return ood, ood_list

def calc_ood_stats(train_path, test_path, th_x, th_q, is_find_ood):
    ood, ood_list = get_ood_list(th_x, th_q, is_find_ood)    
    #test_path = 'features/dfnet_features_test'    
    test_df = pd.read_csv(test_path+'/dfnet_res.csv')    
    test_df = test_df.reset_index()  # make sure indexes pair with number of rows    
    #loop over all poses and find images where test pose if far from all train poses
    high_err_in_ood = 0
    high_err_not_in_ood = 0
    
    od = []
    id = []
    for i1, test_row in test_df.iterrows():
        ang_error = float(test_row['ang_error'])
        x_error = float(test_row['x_error'])
        filename = os.path.basename(test_row['filename'])
        if x_error > th_x or ang_error > th_q:
            if filename in ood_list:
                high_err_in_ood += 1
            else:
                high_err_not_in_ood += 1
        if filename in ood_list:
            od.insert(0, [x_error, ang_error])
        else:
            #id.insert(0, [x_error, ang_error])
            id.append([x_error, ang_error])
    
    od = np.array(od)
    id = np.array(id)
    ttest_result = stats.ttest_ind(id, od)
    print(ttest_result)    
    ttest_result = stats.ttest_ind(id, od, equal_var=False)
    print(ttest_result)    
                    
    print('testset size: ' + str(test_df.shape[0]))
    print('th_x: '+str(th_x)+'m th_q: '+str(th_q)+'deg')
    print('ood size: '+str(len(ood)-1))
    print('high_err_in_ood: ' + str(high_err_in_ood))
    print('high_err_not_in_ood: ' + str(high_err_not_in_ood))              
      

def analyze_ood(train_path, test_path, filename):    
    print('find nn in train for test image: ' + filename)
    print('----------------------------------------------------------------')
    nn_list_x, nn_list_theta = find_nn_by_distance(train_path, test_path, filename)
    n = 5
    n_items = list(nn_list_x.items())[:n]
    print('nn_list_x: ')
    print(n_items)
    n_items = list(nn_list_theta.items())[:n]
    print('nn_list_theta: ')
    print(n_items)
    
    

def main():
    is_find_ood = False
    th_x = 5
    th_q = 10
    thr_sim = 0.8
    train_path = 'features/church/dfnet_features_train'
    test_path = 'features/church/dfnet_features_test'
    #calc_ood_stats(train_path, test_path, th_x, th_q, is_find_ood) 
    #ood, ood_str = get_ood_list(train_path, test_path, th_x, th_q, is_find_ood)    
    #find_ood_by_similarity(train_path, test_path, ood, thr_sim)
    filename = "seq13_frame00050.png"    
    analyze_ood(train_path, test_path, filename)
            

if __name__ == "__main__":
    main()