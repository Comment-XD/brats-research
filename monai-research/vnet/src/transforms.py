import torch
from monai.transforms import Transform, Compose

class SingularChannelToMultiChannelMask(Transform):

    def __call__(self, train_label):
        result = []
        # merge label 2 and label 3 to construct TC
        result.append(torch.logical_or(train_label == 2, train_label == 3))
        # merge labels 1, 2 and 3 to construct WT
        result.append(torch.logical_or(torch.logical_or(train_label == 2, train_label == 3), train_label == 1))
        # label 2 is ET
        result.append(train_label == 2)
        result.append(train_label == 0)
        
        return torch.stack(result, axis=0).float()

def label_transform():
    return Compose([
        SingularChannelToMultiChannelMask()
    ])

