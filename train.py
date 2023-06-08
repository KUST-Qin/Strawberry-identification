import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import tqdm


def train():
    # model init
    model = torchvision.models.vgg16_bn(pretrained=True)
    model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=2, bias=2)

    # load data from dataset
    dataset = torchvision.datasets.ImageFolder(
        root='./data', transform=transforms.Compose([transforms.Resize([224, 224]),
                                                     transforms.ToTensor()]),
        is_valid_file=None)
    # create dataloader
    data_train = DataLoader(dataset, batch_size=4, num_workers=0, drop_last=True, shuffle=True)

    # hyper-params
    learning_rate = 1e-4
    epochs = 40
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    lossl1 = torch.nn.CrossEntropyLoss()
    # start training
    for i in range(epochs):
        data_train_iter = iter(data_train)
        for j in tqdm.tqdm(range(len(data_train))):
            a, b = next(data_train_iter)
            optimizer.zero_grad()
            out = model(a)
            loss = lossl1(out, b)
            loss.backward()
            optimizer.step()
            if (j + 1) % 10 == 0:
                print('Epoch {}  Training loss {:.4f}'.format(i, loss))

        # test every 5 epoch
        if (i + 1) % 5 == 0:
            data_test_iter = data_train.__iter__()
            correct = 0
            for _ in tqdm.tqdm(range(len(data_train))):
                a, b = data_test_iter.next()
                out = model(a)
                _, pred = torch.max(out, dim=1)
                correct += sum(i for i in pred == b if i)
            acc = correct / (len(data_train) * 4)
            print('test acc {:.4f}'.format(acc))

        # save model every 10 epoch
        if (i + 1) % 10 == 0:
            torch.save(model.state_dict(), 'epoch_{}.pth'.format(i + 1))


if __name__ == '__main__':
    train()