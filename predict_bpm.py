import numpy as np
import torch
from nets.models.MTTS_CSTM_Adjust import MTTS_CSTM 

class Prediction_bpm(object):
    def __init__(self):
        self.pop_mean = [
            [0.284967075, 0.344768359, 0.582282681],
            [2.91646956e-07, 4.81673676e-07, 1.42055139e-06]
        ]
        self.pop_std = [
            [0.1052833, 0.13523202, 0.22788221],
            [0.0252121, 0.03166188, 0.0509791]
        ]
        self.frame_depth = 10
        self.shift_factor = 0.625
        self.skip_connection = True
        self.new_group_tsm = False
        self.model = self.init_model()
        
    def init_model(self):
        model = MTTS_CSTM(frame_depth = self.frame_depth, pop_mean=self.pop_mean, pop_std=self.pop_std, shift_factor=self.shift_factor, skip=self.skip_connection, group_on=self.new_group_tsm)
        checkpoint = torch.load('checkpoint_MMSE/MTTS_CSTM_MMSE_T_10_shift_0.625_combined_loss_best_model_1.pth',  map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model"])        
        model.eval() 
        return model
        
   
    def get_inputs(self, arr_app, arr_motion):
        new_arr_app = arr_app[:10]
        tensor_app = torch.tensor(new_arr_app)
        tensor_motion = torch.tensor(arr_motion)
        combined_tensor = torch.stack([ tensor_motion, tensor_app], dim=0)
        inputs = combined_tensor.permute(0, 1, 4, 2, 3).unsqueeze(0)
        # print('inputs: ', inputs)
        return inputs
    
    
    def predict_bpm(self, arr_app, arr_motion):
        inputs = self.get_inputs(arr_app, arr_motion)
        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs