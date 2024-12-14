import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision import transforms

class MTTSDataset_Hao(Dataset):
    def __init__(self, file, ds_name, window_length, valid=False, ImgROI=None):
        super(MTTSDataset_Hao, self).__init__()
        self.ds_name = ds_name
        self.transform = torch.nn.Sequential(
            transforms.Resize((ImgROI, ImgROI))
        )
        self.size = ImgROI
        self.valid = valid
        self.window_length = window_length

        if type(self.ds_name) == list:
            self.fs=[]
            self.video_fs=[]
        self._get_arrays(file)
        self.pre_train = True

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
       
        wl = self.window_length
        if type(self.ds_name) == list:
            if idx+wl > self.total_length-1:   # idx overpass total_length then idx backward to wl-total_length+1
                idx -= idx + wl - self.total_length + 1   
            
        else:
            idx = int(idx * self.window_length) # it means idx += 10(nonoverlapping)
            
        x = torch.tensor(self.video[idx:idx + self.window_length + 1], dtype=torch.float32).permute(0, 3, 1, 2)
        y = torch.squeeze(torch.tensor(self.label[idx:idx + self.window_length], dtype=torch.float32))
        x = self.transform(x)
        motion_frames = torch.empty((self.window_length, 3, self.size, self.size), dtype=torch.float32)
        for i in range(self.window_length):
            motion_frames[i] = x[i + 1] - x[i]

        average_frame = x[:-1]      # Dao
       
        x = torch.stack((motion_frames, average_frame))  # -> 2, T, 36, 36, 3
        return x, y

    def _get_arrays(self, file):
        print("file: ", file)
        if type(self.ds_name) == list:
            for t, f in enumerate(file):
                with tqdm(total=len(list(file[t].keys())), position=0, leave=True, desc='Reading from file') as pbar:
                    self.n_frames_per_video = np.empty((len(list(file[t].keys()))), dtype=np.int_)
                    for i, data_path in enumerate(list(file[t].keys())):
                        n_frames_per_video = len(file[t][data_path]['label'])
                        self.n_frames_per_video[i] = n_frames_per_video
                        if self.ds_name[t] == "UBFC" or "PURE":   # change here
                            self.fs.extend([30] * n_frames_per_video)  # change the fs of window
                            self.video_fs.append(30)
                        elif self.ds_name[t] == "MMSE":  # change here
                            self.fs.extend([25] * n_frames_per_video)   # change the fs
                            self.video_fs.append(25)  # fs of video
                        elif self.ds_name[t] == "MANHOB_HCI":  # change here
                            self.fs.extend([61] * n_frames_per_video)   # change the fs
                            self.video_fs.append(61)  # fs of video
                        video_frames = file[t][data_path]['video']
                        labels = file[t][data_path]['label']
                        if i == 0 and t == 0:
                            self.video = video_frames
                            self.label = labels
                        else:
                            self.video = np.append(self.video, video_frames, axis=0)
                            self.label = np.append(self.label, labels)
                        pbar.update(1)
            self.total_length = (len(self.label) - 1) # // self.window_length
            print(f"self.total_length={self.total_length}")
        else:
            with tqdm(total=len(list(file.keys())), position=0, leave=True, desc='Reading from file') as pbar:
                self.n_frames_per_video = np.empty((len(list(file.keys()))), dtype=int)
                for i, data_path in enumerate(list(file.keys())):
                    n_frames_per_video = len(file[data_path]['label'])
                    self.n_frames_per_video[i] = n_frames_per_video
                    video_frames = file[data_path]['video']
                    labels = file[data_path]['label']
                    if i == 0:
                        self.video = video_frames
                        self.label = labels
                    else:
                        self.video = np.append(self.video, video_frames, axis=0)
                        self.label = np.append(self.label, labels)
                    pbar.update(1)
            self.total_length = (len(self.label) - 1) // self.window_length
            print(f"self.total_length={self.total_length}")

    def update_state(self):
        self.pre_train = False
        self.total_length = (self.total_length * self.window_length) - self.window_length

