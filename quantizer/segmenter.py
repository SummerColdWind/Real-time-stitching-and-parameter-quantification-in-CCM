import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # , utils
from pathlib import Path
from skimage import io

from .u2net.data_loader import RescaleT
from .u2net.data_loader import ToTensorLab
from .u2net.data_loader import SalObjDataset

from .u2net.model import U2NETP

checkpoint = os.path.join(os.path.dirname(__file__), 'u2netp_ccm_plus.pth')

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


class Segmenter:
    def __init__(self):
        self.model = U2NETP(3, 1)
        self.model.load_state_dict(torch.load(checkpoint))
        self.model.cuda()
        self.model.eval()

        self.inputs = []


    def preprocess(self, inputs):
        self.inputs.clear()
        for x in inputs:
            if isinstance(x, (str, Path)):
                x = io.imread(x)
            self.inputs.append(x)

        test_salobj_dataset = SalObjDataset(img_name_list=self.inputs.copy(),
                                            lbl_name_list=[],
                                            transform=transforms.Compose([RescaleT(320),
                                                                          ToTensorLab(flag=0)])
                                            )
        test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=0)

        return test_salobj_dataloader

    @staticmethod
    def postprocess(pred, size):
        pred = pred.squeeze()
        predict_np = pred.cpu().data.numpy()
        gray = (predict_np * 255).astype('uint8')
        gray = cv2.resize(gray, size)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def forward(self, input_):
        preds = []
        dataloader = self.preprocess(input_)
        for i, x in enumerate(dataloader):
            h, w = self.inputs[i].shape[:2]
            inputs_test = x['image']
            inputs_test = inputs_test.type(torch.FloatTensor)

            inputs_test = Variable(inputs_test.cuda())

            d1, d2, d3, d4, d5, d6, d7 = self.model(inputs_test)

            # normalization
            pred = d1[:, 0, :, :]
            pred = normPRED(pred)

            del d1, d2, d3, d4, d5, d6, d7

            preds.append(self.postprocess(pred, (w, h)))
        return preds

    def __call__(self, inputs: list):
        return self.forward(inputs)
