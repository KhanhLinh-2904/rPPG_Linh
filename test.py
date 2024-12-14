
import torch
import math
from loss2 import loss_fn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from dataset_loader2 import dataset_loader
from funcs2 import BPF_dict, normalize
import numpy as np 
from nets.models.MTTS_CSTM_Adjust import MTTS_CSTM     # Adjust the model fit different image_ROI

def main():
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    model_name = "MTTS_CSTM"

    # test
    save_root_path = "MMSE_test/"        # Dataset size = 36*36   
   
    checkpoint_path = "checkpoint_MMSE/"

    print("checkpoint_path: ", checkpoint_path)
    dataset = [["MMSE"]]
    fs = [25]
    Test_SW_list = [10]                 # for Test Sliding Windows 
    for Test_SW in Test_SW_list:
        print("----Now Test Slinding Window is setting by {}----".format(Test_SW))
        for fs_cnt, dataset_name in enumerate(dataset):
            for TT in range(1, 6):
                batch_size = 32
                loss_metric = "combined_loss"  # combined_loss  mse
                
                window_length = 10          # Defalut: 10, 20
                shift_factor = 0.625         # Defalut: 0.25

                ROI = 36                            # Adjust faceROI to 36 | 54 | 72
                
                skip_connection = True      # True: Residual  False: In-place
                new_group_tsm = False
                checkpoint_name = "MTTS_CSTM_" + dataset_name[0] + "_T_"+ str(window_length) + "_shift_" + str(shift_factor) + "_" + loss_metric + "_best_model_" + str(TT) + ".pth" # FiveFold v1


                    
                test_dataset = dataset_loader(2, save_root_path, model_name, dataset_name, window_length, fold=TT, SW=Test_SW, ImgROI=ROI, is_test=True)
                print(f"length of test dataset is {len(test_dataset)}")
                test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=SequentialSampler(test_dataset),
                                            num_workers=6, pin_memory=True, drop_last=False)
                print(f"The name of checkpoint is {checkpoint_name}")
                app_mean = []
                app_std = []
                motion_mean = []
                motion_std = []

              
                with tqdm(total=len(test_dataset), position=0, leave=True,
                        desc='Calculating population statistics') as pbar:

                    for data in test_loader:
                        if model_name in ['TSDAN', 'MTTS', 'MTTS_CSTM']:
                            data = data[0]  # -> (Batch, 2, T, H, W, 3)
                            # print(f"data.shape={data.shape}")
                            # data.shape=torch.Size([32, 2, 10, 3, 36, 36])
                            motion_data, app_data = torch.tensor_split(data, 2, dim=1)
                            B, one, T, C, H, W = motion_data.shape

                            motion_data = motion_data.view(B*one, T, C, H, W)
                            app_data = app_data.view(B*one, T, C, H, W)
                            motion_data = motion_data.reshape(B*T, C, H, W)
                            app_data = app_data.reshape(B*T, C, H, W)

                            batch_motion_mean = torch.mean(motion_data, dim=(0, 2, 3)).tolist()
                            batch_motion_std = torch.std(motion_data, dim=(0, 2, 3)).tolist()
                            batch_app_mean = torch.mean(app_data, dim=(0, 2, 3)).tolist()
                            batch_app_std = torch.std(app_data, dim=(0, 2, 3)).tolist()

                            app_mean.append(batch_app_mean)
                            app_std.append(batch_app_std)
                            motion_mean.append(batch_motion_mean)
                            motion_std.append(batch_motion_std)

                        pbar.update(B)

                    pbar.close()

                if model_name in ['TSDAN', 'MTTS', 'MTTS_CSTM', 'SlowFast_FD']:
                    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
                    app_mean = np.array(app_mean).mean(axis=0) / 255
                    app_std = np.array(app_std).mean(axis=0) / 255
                    motion_mean = np.array(motion_mean).mean(axis=0) / 255
                    motion_std = np.array(motion_std).mean(axis=0) / 255
                    pop_mean = np.stack((app_mean, motion_mean))  # 0 is app, 1 is motion
                    pop_std = np.stack((app_std, motion_std))

                print(" pop_mean: ",pop_mean )
                print(" pop_std: ",pop_std )
               
                model = MTTS_CSTM(frame_depth = window_length, pop_mean=pop_mean, pop_std=pop_std, shift_factor=shift_factor, skip=skip_connection, group_on=new_group_tsm)

                criterion = loss_fn(loss_metric)

                min_val_loss = 10000

                valid_loss = []

                print("checkpoint_path + checkpoint_name: ",checkpoint_path + checkpoint_name)
                checkpoint = torch.load(checkpoint_path + checkpoint_name,  map_location=torch.device('cpu'))
                model.load_state_dict(checkpoint["model"])
                valid_loss = checkpoint["valid_loss"]
                checkpoint_epoch = checkpoint['epoch']
                min_val_loss = valid_loss[-1]
                print(f"epoch={checkpoint_epoch}, min_val_loss={min_val_loss}")

                with tqdm(test_loader, desc="Validation ", total=len(test_loader), colour='green') as tepoch:
                    model.eval()    # Model for evaluation
                    running_loss = 0.0

                    inference_array = []
                    target_array = []
                    inference_array_avg=[]
                    target_array_avg=[]

                    with torch.no_grad():   # Regression has been forbidden - no gradient
                       
                        for inputs, target in tepoch:
                            tepoch.set_description(f"Test")
                            if torch.isnan(target).any():
                                print('A')
                                return
                            if torch.isinf(target).any():
                                print('B')
                                return

                            print("================inputs: ", inputs.shape)
                            outputs = model(inputs)
                            if torch.isnan(outputs).any():
                                print('A')
                                return
                            if torch.isinf(outputs).any():
                                print('B')
                                return
                            loss = criterion(outputs, target)

                            running_loss += loss.item() * target.size(0) * target.size(1)
                            tepoch.set_postfix(loss='%.6f' % (running_loss / len(test_loader) / window_length /batch_size))
                            inference_array = np. append(inference_array, np.reshape(outputs.cpu().detach().numpy(), (1, -1)))
                            target_array = np.append(target_array, np.reshape(target.cpu().detach().numpy(), (1, -1)))

                        valid_loss.append(running_loss / len(test_loader) / window_length /batch_size)

                Step = window_length//Test_SW
                inference_len = len(inference_array)
                WL = window_length
             
                #-- the begining step --#
                for i in range(0, WL, 1):
                    sum_value_inf = 0
                    sum_value_tar = 0
                    for index in range(0,i//Test_SW+1,1):
                        sum_value_inf = sum_value_inf + inference_array[(index*(WL-Test_SW))+i]
                        sum_value_tar = sum_value_tar + target_array[(index*(WL-Test_SW))+i]
                    inference_array_avg.append(sum_value_inf/(i//Test_SW+1))
                    target_array_avg.append(sum_value_tar/(i//Test_SW+1))                

                #-- the middle step --#
                for i in range((2*WL)-Test_SW, inference_len-((Step-1)*WL), WL):
                    for j in range(0, Test_SW, 1):
                        sum_value_inf = 0
                        sum_value_tar = 0
                        for index in range(0,Step,1):
                            sum_value_inf = sum_value_inf + inference_array[(index*(WL-Test_SW))+i+j]
                            # sum_value_inf = sum_value_inf + inference_array[(index*10)+j]
                            sum_value_tar = sum_value_tar + target_array[(index*(WL-Test_SW))+i+j]
                        inference_array_avg.append(sum_value_inf/Step)
                        target_array_avg.append(sum_value_tar/Step)

                #-- the end step --#
                for i in range(-WL+Test_SW, 0):
                    sum_value_inf = 0
                    sum_value_tar = 0
                   
                    avg_number = math.ceil(abs(i)/Test_SW)
                    for index in range(0,avg_number,1):
                        position = i-(index*(WL-Test_SW))
                        sum_value_inf = sum_value_inf + inference_array[position]
                        sum_value_tar = sum_value_tar + target_array[position]              
                    inference_array_avg.append(sum_value_inf/avg_number)
                    target_array_avg.append(sum_value_tar/avg_number)     

              
                #---------#
                result = {}
                groundtruth = {}
                start_idx = 0
                n_frames_per_video = test_dataset.n_frames_per_video   # show the total frame of each video 
               
                for i, value in enumerate(n_frames_per_video):
                    
                    result[i] = normalize(inference_array_avg[start_idx:start_idx + value])
                    groundtruth[i] = target_array_avg[start_idx:start_idx + value]
                    start_idx += value

                result = BPF_dict(result, fs[fs_cnt])
                groundtruth = BPF_dict(groundtruth, fs[fs_cnt])
                print('result: ', result)
                print('groundtruth: ', groundtruth)

              
if __name__ == '__main__':
    print("************************************************************************************************")
    main()
    print("************************************************************************************************")
