import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from models import LinkNet34
from torch.autograd import Variable
from PIL import Image



class FaceSegmentation(object):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = LinkNet34()
        self.model.load_state_dict(torch.load('linknet.pth', map_location=lambda storage, loc: storage))
        self.model.eval()
        self.model.to(self.device)

    def face_segment(self, face_frame):
        img_transform = transforms.Compose([transforms.Resize((255, 255)), transforms.ToTensor(), transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        shape = face_frame.shape
        if len(shape) != 3:
            frame = cv2.cvtColor(face_frame, cv2.COLOR_GRAY2RGB)
        else:
            frame = face_frame
        frame = cv2.resize(frame, (255, 255), cv2.INTER_LINEAR)
        a = img_transform(Image.fromarray(frame))
        a = a.unsqueeze(0)
        imgs = Variable(a.to(dtype=torch.float, device=self.device))
        pred = self.model(imgs)
        pred = torch.nn.functional.interpolate(pred, size=[shape[0], shape[1]])
        mask = pred.data.cpu().numpy()
        mask = mask.squeeze()

        mask = mask > 0.9  # todo: optimal value with 25 cm from the camera
        if len(shape) != 3:
            rgba = cv2.cvtColor(face_frame, cv2.COLOR_GRAY2RGBA)
        else:
            rgba = cv2.cvtColor(face_frame, cv2.COLOR_RGB2RGBA)
        ind = np.where(mask == 0)
        black_point = len(ind[0])
        rgba[ind] = [0, 0, 0, 255]  # fill background with pit black

        canvas = Image.new('RGBA', (rgba.shape[1], rgba.shape[0]), (255, 255, 255, 255))
        canvas.paste(Image.fromarray(rgba), mask=Image.fromarray(rgba))
        rgba = np.array(canvas)
        if len(shape) != 3:
            rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
        else:
            rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
        return rgb,black_point
