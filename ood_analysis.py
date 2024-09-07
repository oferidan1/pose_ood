import os
import time
import torch
import faiss
import numpy as np
from glob import glob
from faiss_index import LocalFaissIndex
import pandas as pd
import math
from scipy import stats
import argparse
from eigenplaces_model import eigenplaces_network
from PIL import Image
import torchvision.transforms as transforms
import torchvision
from util import utils, bf_matching
import cv2

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def get_pose_diff_from_csv(train_pose_x, train_pose_q, test_pose_x, test_pose_q):    
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
            train_pose_x = train_row['gt_pose_x']
            train_pose_q = train_row['gt_pose_q']
            train_pose_x = torch.from_numpy(np.array([float(x.strip(' []')) for x in train_pose_x.split(',')]))
            train_pose_q = torch.from_numpy(np.array([float(x.strip(' []')) for x in train_pose_q.split(',')]))
            error_x, theta = get_pose_diff_from_csv(train_pose_x, train_pose_q, test_pose_x, test_pose_q)
            if error_x > th_x or theta > th_q:                
                cnt_far_neighbors += 1
        if cnt_far_neighbors == train_df.shape[0]:
            ood_list.append(test_row['filename'])
            
    print(len(ood_list))
    return ood_list           

def find_nn_by_distance(train_path, test_path, test_image, thr_x, thr_q):      
    train_df = pd.read_csv(train_path+'/dfnet_res.csv')
    test_df = pd.read_csv(test_path+'/dfnet_res.csv')
    train_df = train_df.reset_index()  # make sure indexes pair with number of rows
    test_df = test_df.reset_index()  # make sure indexes pair with number of rows
    nn_list_x = {}
    nn_list_theta = {}
    nn_list_closest = {}
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
            train_pose_x = train_row['gt_pose_x']
            train_pose_q = train_row['gt_pose_q']
            train_pose_x = torch.from_numpy(np.array([float(x.strip(' []')) for x in train_pose_x.split(',')]))
            train_pose_q = torch.from_numpy(np.array([float(x.strip(' []')) for x in train_pose_q.split(',')]))
            error_x, theta = get_pose_diff_from_csv(train_pose_x, train_pose_q, test_pose_x, test_pose_q)      
            train_filename = train_row['filename']
            if error_x < thr_x and theta < thr_q:
                nn_list_closest[train_filename] = [error_x, theta]
            if error_x < min_error_x:
                min_error_x = error_x
                nn_list_x[train_filename] = min_error_x
            if theta < min_theta:
                min_theta = theta
                nn_list_theta[train_filename] = min_theta
                      
        
    nn_list_x = dict(sorted(nn_list_x.items(), key=lambda item: item[1]))
    nn_list_theta = dict(sorted(nn_list_theta.items(), key=lambda item: item[1]))
    return nn_list_x, nn_list_theta, nn_list_closest


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
    
    
def find_nn_by_similarity(train_path, test_path, testname=None):
    #load train features to faiss    
    #print('test file: ' + testname)
    is_normalize = True
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
        
    #go over test images and find best match in train    
    i = 0
    k = 20
    test_train_nn = {}
    for name in test_names:
        if testname is not None and testname not in name:
            i += 1        
            continue        
        feat = ftest[i]        
        feat = np.expand_dims(feat, axis=0)
        if is_normalize:                    
            scores, idx = train_faiss.search(feat, k)       
            nn_name = train_names[idx[0][0]]
            nn_score = scores[0][0]
        else:
            idx, scores, names = train_faiss.search(feat, k)               
            nn_name = name[0]
            nn_score = scores[0]
        test_train_nn[name] = {'nn': nn_name, 'score': nn_score}
        i += 1        
        if testname is not None:
            break
    
    if testname is not None:
        print('train nn: ')
        print(names)
        print('train nn scores: ')
        print(scores)    
        
    #print(test_train_nn)
    
    return test_train_nn
      

def analyze_ood(train_path, test_path, filename, thr_x, thr_q):    
    print('find nn in train for test image: ' + filename)
    print('----------------------------------------------------------------')
    nn_list_x, nn_list_theta, nn_list_closest = find_nn_by_distance(train_path, test_path, filename,  thr_x, thr_q)
    n = 5
    n_items = list(nn_list_x.items())[:n]
    print('nn_list_x: ')
    print(n_items)
    n_items = list(nn_list_theta.items())[:n]
    print('nn_list_theta: ')
    print(n_items)
    print('nn_list_closest: ')
    print(nn_list_closest)
    
def load_images(test_img_name, train_img_name, args):    
    base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    reproj_transform = transforms.Compose([
            transforms.Resize((270, 480)),
            transforms.ToTensor(),
        ])

    test_img = Image.open(test_img_name)
    test_img = base_transform(test_img).unsqueeze(0)
    train_img = Image.open(train_img_name)
    train_img = base_transform(train_img).unsqueeze(0)
    eigen_images = torch.cat((test_img, train_img), 0)
    
    #reproj
    test_img_270 = Image.open(test_img_name)
    test_img_270 = reproj_transform(test_img_270).unsqueeze(0)
    train_img_270 = Image.open(train_img_name)
    train_img_270 = reproj_transform(train_img_270).unsqueeze(0)    
    filename = os.path.splitext(os.path.basename(train_img_name))[0]   
    bTrainDepthFound = True 
    depth_file_name = args.dataset_folder + 'depth_noseg/' + filename + '.depth.tiff'
    if not os.path.exists(depth_file_name):
        depth_file_name = args.dataset_folder + 'depth_noseg/' + filename + '.depth.png'
        if not os.path.exists(depth_file_name):            
            print('missing: ' + depth_file_name)
            bTrainDepthFound = False        
            train_img_depth = None
    if bTrainDepthFound:
        img_depth = cv2.imread(depth_file_name, -1)
        img_depth = cv2.resize(img_depth, (480, 270))    
        train_img_depth = torch.from_numpy(img_depth.astype(np.float32)).unsqueeze(0).unsqueeze(0)    
    filename = os.path.splitext(os.path.basename(test_img_name))[0]        
    depth_file_name = args.dataset_folder + 'depth_noseg/' + filename + '.depth.tiff'    
    if not os.path.exists(depth_file_name):
        print('missing: ' + depth_file_name)
        test_img_depth = None
    else:
        img_depth = cv2.imread(depth_file_name, -1)    
        img_depth = cv2.resize(img_depth, (480, 270))    
        test_img_depth = torch.from_numpy(img_depth.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return eigen_images, test_img_270, train_img_270, test_img_depth, train_img_depth
    
def analyze_overlap(train_path, test_path, args, out_csv):
    test_train_nn = find_nn_by_similarity(train_path, test_path)
    train_df = pd.read_csv(train_path+'/dfnet_res.csv')
    test_df = pd.read_csv(test_path+'/dfnet_res.csv')
    train_df = train_df.reset_index()  # make sure indexes pair with number of rows
    test_df = test_df.reset_index()  # make sure indexes pair with number of rows    
    #loop over all poses and find images where test pose if far from all train poses
    test_train_res = []
    eigen_model = eigenplaces_network.GeoLocalizationNet_(args.backbone, args.fc_output_dim)
    eigen_model = eigen_model.to(args.device)
    eigen_model = eigen_model.eval()
    mse = torch.nn.MSELoss()
    for i1, test_row in test_df.iterrows():
        test_pose_x = test_row['gt_pose_x']
        test_pose_q = test_row['gt_pose_q']
        test_pose_x = torch.from_numpy(np.array([float(x.strip(' []')) for x in test_pose_x.split(',')]))
        test_pose_q = torch.from_numpy(np.array([float(x.strip(' []')) for x in test_pose_q.split(',')]))
        testname = os.path.splitext(os.path.basename(test_row['filename']))[0]
        train_nn = test_train_nn[testname]
        for i2, train_row in train_df.iterrows():
            if train_nn['nn'] in train_row['filename']:
                #compute pose diff
                train_pose_x = train_row['gt_pose_x']
                train_pose_q = train_row['gt_pose_q']
                train_pose_x = torch.from_numpy(np.array([float(x.strip(' []')) for x in train_pose_x.split(',')]))
                train_pose_q = torch.from_numpy(np.array([float(x.strip(' []')) for x in train_pose_q.split(',')]))
                error_x, theta = get_pose_diff_from_csv(train_pose_x, train_pose_q, test_pose_x, test_pose_q)                
                
                #compute eigen places similarity
                test_img = args.test_images_dir + testname + '.png'
                train_img = args.train_images_dir + train_nn['nn'] + '.png'
                eigen_images, test_img_270, train_img_270, test_img_depth, train_img_depth = load_images(test_img, train_img, args)                
                with torch.no_grad():
                    eigen_descriptors = eigen_model(eigen_images.to(args.device))
                eigen_sim = eigen_descriptors[0]@eigen_descriptors[1].t()
                eigen_sim = eigen_sim.detach().cpu().numpy()
                
                #compute reproj loss
                train_pose = torch.cat((train_pose_x, train_pose_q), 0).unsqueeze(0)
                test_pose = torch.cat((test_pose_x, test_pose_q), 0).unsqueeze(0)
                reproj_img = utils.reproject_RGB(train_img_270, train_img_depth, train_pose, test_pose, True)
                #torchvision.utils.save_image(reproj_img, 'train_reproj.png')
                non_zero_mask = torch.where(reproj_img[0,0] > 0, 1, 0) & torch.where(reproj_img[0,1] > 0, 1, 0) & torch.where(reproj_img[0,2] > 0, 1, 0)
                cnt_non_zero = non_zero_mask.sum()
                b,c,h,w = reproj_img.shape
                reproj_overlap_train = float((cnt_non_zero / (h*w)).numpy())
                reproj_overlap_mse_train = float(mse(reproj_img * non_zero_mask, test_img_270))#/cnt_non_zero
                reproj_overlap_test = 0
                reproj_overlap_mse_test = 0
                if test_img_depth is not None:
                    reproj_img = utils.reproject_RGB(test_img_270, test_img_depth, test_pose, train_pose, True)
                    non_zero_mask = torch.where(reproj_img[0,0] > 0, 1, 0) & torch.where(reproj_img[0,1] > 0, 1, 0) & torch.where(reproj_img[0,2] > 0, 1, 0)
                    cnt_non_zero = non_zero_mask.sum()
                    reproj_overlap_test = float((cnt_non_zero / (h*w)).numpy())
                    reproj_overlap_mse_test = float(mse(reproj_img * non_zero_mask, train_img_270))#/cnt_non_zero                
                
                #keep image stats
                test_train_res.append({'test_name': testname, 'error_x':round(test_row['x_error'],3), 'error_q':round(test_row['ang_error'],3), 'nn': train_nn['nn'], 
                                       'nn_score': round(train_nn['score'],3), 'nn_error_x':round(float(error_x),3), 'nn_error_q': round(float(theta),3), 'nn_eigen': round(float(eigen_sim),3),
                                       'reproj_overlap_train': round(reproj_overlap_train, 3), 'reproj_L2_train': round(reproj_overlap_mse_train, 3),
                                       'reproj_overlap_test': round(reproj_overlap_test, 3), 'reproj_L2_test': round(reproj_overlap_mse_test, 3)})
                
    df = pd.DataFrame(test_train_res)
    df.to_csv(out_csv, index=False)
    return test_train_res


def analyze_nn_match(scene_folder, nn_csv):
    nn_df = pd.read_csv(nn_csv)    
    nn_df = nn_df.reset_index()  # make sure indexes pair with number of rows    
    train_sp_folder = scene_folder + '/train/superpoint/'
    test_sp_folder = scene_folder + '/test/superpoint/'
    ratio_thresh = 0.7
    good_matches_cnt = []
    for i1, row in nn_df.iterrows():
        test_sp_name = test_sp_folder + row['test_name'] + '.npy'
        train_sp_name = train_sp_folder + row['nn'] + '.npy'
        test_sp = np.load(test_sp_name, allow_pickle=True)
        #keypoints = test_sp.item().get('keypoints')[0]
        #scores = test_sp.item().get('scores')[0]
        test_desc = test_sp.item().get('descriptors')[0].t()
        train_sp = np.load(train_sp_name, allow_pickle=True)
        train_desc = train_sp.item().get('descriptors')[0].t()
        knn_matches = bf_matching.match_descriptors(test_desc.detach().cpu().numpy(), train_desc.detach().cpu().numpy())
        #-- Filter matches using the Lowe's ratio test
        good_matches = []
        good = 0
        for m,n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
                good += 1
        good_matches_cnt.append(good)
    
    df2 = nn_df.assign(good_match=good_matches_cnt)
    df2.to_csv(nn_csv, index=False)
    return good_matches_cnt


def parse_arguments():
    parser = argparse.ArgumentParser("ood")
    # CosPlace Groups parameters
    # Model parameters
    parser.add_argument("--backbone", type=str, default="ResNet18", choices=["VGG16", "ResNet18", "ResNet50", "ResNet101", "ResNet152"], help="_")
    parser.add_argument("--fc_output_dim", type=int, default=512,  help="Output dimension of final fully connected layer")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="_")
    parser.add_argument("--scene", type=str, default="shop", help="_")
    parser.add_argument("--dataset_folder", type=str, default="/mnt/f/CambridgeLandmarks/ShopFacade/", help="_")
    args = parser.parse_args()
    return args
    
def main(args):
    is_find_ood = False
    th_x = 1
    th_q = 40
    thr_sim = 0.8
    train_path = 'features/' + args.scene + '/dfnet_features_train'
    test_path = 'features/' + args.scene + '/dfnet_features_test'
    #calc_ood_stats(train_path, test_path, th_x, th_q, is_find_ood) 
    #ood, ood_str = get_ood_list(train_path, test_path, th_x, th_q, is_find_ood)    
    #find_ood_by_similarity(train_path, test_path, ood, thr_sim)
    filename = "seq1_frame00005"    
    #analyze_ood(train_path, test_path, filename, th_x, th_q)
    #find_nn_by_similarity(train_path, test_path, filename)    
    args.train_images_dir = args.dataset_folder + '/train/rgb/'
    args.test_images_dir = args.dataset_folder + '/test/rgb/'
    out_csv = args.scene + '_overlap.csv'
    #analyze_overlap(train_path, test_path, args, out_csv)
    good_matches_cnt = analyze_nn_match(args.dataset_folder, out_csv)
    
            

if __name__ == "__main__":
    args = parse_arguments()
    main(args)