import torch
import os
import csv

def load_tensor_from_file(file_path):
    tensor = torch.load(file_path,map_location=torch.device('cpu'))
    #remove 1 dimension
    tensor = tensor.squeeze()
    tensor = tensor / tensor.norm(dim=-1, keepdim=True)
    return tensor

def load_tensor_from_file_visual(file_path):
    tensor = torch.load(file_path,map_location=torch.device('cpu'))
    tensor = tensor.squeeze()
    tensor = tensor / tensor.norm(dim=-1, keepdim=True)
    return tensor

def load_tensor_from_file_mask(file_path):
    tensor = torch.load(file_path,map_location=torch.device('cpu'))
    tensor = tensor.squeeze()
    return tensor

def _mean_pooling_for_similarity_visual(visual_output, video_mask,):
    video_mask = video_mask.view(-1, video_mask.shape[-1])
    video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
    visual_output = visual_output * video_mask_un
    video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
    video_mask_un_sum[video_mask_un_sum == 0.] = 1.
    video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
    video_out = video_out / video_out.norm(dim=-1, keepdim=True)
    #test if the norm is 1
    
    
    return video_out

#manipulation = ['temporal_int', 'temporal_act', 'neighborhood_same_entity', 'neighborhood_diff_entity', 'counter_rel', 'counter_act', 'counter_int', 'counter_attr']
manipulation = ['temporal_contact_swap','temporal_action_swap','neighborhood_same_entity','neighborhood_diff_entity','counter_spatial','counter_contact','counter_action','counter_attribute']

seed = [2,3,42]

pos_text2video_result = {}
neg_text2video_result = {}
pos_text2neg_text_result = {}
for j in range(len(seed)):
    seed_num = seed[j]
    for i in range(len(manipulation)):
        experiment = manipulation[i] 
        #load the data
        text_feature_path = '/home/wiss/zhang/nfs/video_prober/xclip/anetqa/'+experiment+'/batch_sequence_output_list.'+str(seed_num)
        video_feature_path = '/home/wiss/zhang/nfs/video_prober/xclip/anetqa/'+experiment+'/video_representations.'+str(seed_num)
        video_mask_path = '/home/wiss/zhang/nfs/video_prober/xclip/anetqa/'+experiment+'/batch_list_v.'+str(seed_num)
        neg_text_feature_path = '/home/wiss/zhang/nfs/video_prober/xclip/anetqa/'+experiment+'_mani/batch_sequence_output_list.'+str(seed_num)
        
        save_path = '/home/wiss/zhang/nfs/video_prober/xclip/paper_results/'+'anetqa_seed'+str(seed_num)

        text_feature = load_tensor_from_file(text_feature_path)
        video_feature = load_tensor_from_file_visual(video_feature_path)
        video_mask = load_tensor_from_file_mask(video_mask_path)
        neg_text_feature = load_tensor_from_file(neg_text_feature_path)
        
        video_feature_meanp = _mean_pooling_for_similarity_visual(video_feature, video_mask)
        print('text_feature.shape: ', text_feature.shape)
        print('video_feature.shape: ', video_feature.shape)
        print('neg_text_feature.shape: ', neg_text_feature.shape)
        print('video_mask.shape: ', video_mask.shape)
        print('video_feature_meanp.shape: ', video_feature_meanp.shape)

        #get similarity
        pos_t2v_sim = torch.matmul(text_feature, video_feature_meanp.T)
        neg_t2v_sim = torch.matmul(neg_text_feature, video_feature_meanp.T)
        pos_t2neg_t_sim = torch.matmul(text_feature, neg_text_feature.T)
        print('pos_t2v_sim.shape: ', pos_t2v_sim.shape)
        print('neg_t2v_sim.shape: ', neg_t2v_sim.shape)
        print('pos_t2neg_t_sim.shape: ', pos_t2neg_t_sim.shape)
        
        #get the mean value and var value in diagonal
        pos_text2video_mean = pos_t2v_sim.diag().mean()
        pos_text2video_var = pos_t2v_sim.diag().var()
        #save the result into a dict, the key is the manipulation name,the subkey is the mean and var
        pos_text2video_result[experiment] = {'pos_text2video_mean':pos_text2video_mean, 'pos_text2video_var':pos_text2video_var}
        
        neg_text2video_mean = neg_t2v_sim.diag().mean()
        neg_text2video_var = neg_t2v_sim.diag().var()
        #save the result into a dict, the key is the manipulation name,the subkey is the mean and var
        neg_text2video_result[experiment] = {'neg_text2video_mean':neg_text2video_mean, 'neg_text2video_var':neg_text2video_var}
        
        pos_text2neg_text_mean = pos_t2neg_t_sim.diag().mean()
        pos_text2neg_text_var = pos_t2neg_t_sim.diag().var()
        #save the result into a dict, the key is the manipulation name,the subkey is the mean and var
        pos_text2neg_text_result[experiment] = {'pos_text2neg_text_mean':pos_text2neg_text_mean, 'pos_text2neg_text_var':pos_text2neg_text_var}

        #check if the save path exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        #save the result into a csv file
        with open(save_path+'/pos_text2video_result.csv', 'w') as f:
            writer = csv.writer(f)
            for k, v in pos_text2video_result.items():
                writer.writerow([k, v])
        with open(save_path+ '/neg_text2video_result.csv', 'w') as f:
            writer = csv.writer(f)
            for k, v in neg_text2video_result.items():
                writer.writerow([k, v])
        with open(save_path+ '/pos_text2neg_text_result.csv', 'w') as f:
            writer = csv.writer(f)
            for k, v in pos_text2neg_text_result.items():
                writer.writerow([k, v])
    #clear the dict
    pos_text2video_result.clear()
    neg_text2video_result.clear()
    pos_text2neg_text_result.clear()


