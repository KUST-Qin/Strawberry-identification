import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import tqdm
from torchvision.utils import save_image
import cv2
import numpy as np


def train():
    # load model from checkpoint
    model = torchvision.models.vgg16_bn(pretrained=False)
    model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=2, bias=2)
    
    # here
    checkpoint = torch.load('epoch_40.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

    # load test dataset
    dataset = torchvision.datasets.ImageFolder(
        root='./data', transform=transforms.Compose([transforms.Resize([224, 224]),
                                                     transforms.ToTensor()]),
        is_valid_file=None)
    data_test = DataLoader(dataset, batch_size=1, shuffle=True)

    # print 10 inferences
    data_train_iter = iter(data_test)
    # 修改10为其他数字就行
    for i in tqdm.tqdm(range(10)):
        a, b = next(data_train_iter)
        out = model(a)
        _, pred = torch.max(out, dim=1)
        # whether positive or negative
        if pred.item() == 1:
            text = 'good'
        else:
            text = 'damaged'
        src = np.array(a.squeeze(0).permute(1, 2, 0) * 255, dtype=np.uint8)
        src = src[:, :, [2, 1, 0]]
        AddText = src.copy()
        cv2.putText(AddText, text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 100))
        cv2.imwrite('test_{}.jpg'.format(i), AddText)


if __name__ == '__main__':
    train()
