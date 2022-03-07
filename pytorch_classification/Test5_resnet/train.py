import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnet50



def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    with open("a.txt", "a") as f:
        f.write("using {} device.\n".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../../../data"))  # get data root path
    # image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path

    # PlantDoc-Dataset
    # PlantVillage-3_1
    image_path = os.path.join(data_root, "data_set", "PlantDoc-Dataset")  # [可修改]
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 512 # [可修改]
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    with open("a.txt", "a") as f:
        f.write('Using {} dataloader workers every process\n'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    with open("a.txt", "a") as f:
        f.write("using {} images for training, {} images for validation.\n".format(train_num, val_num))

    # PlantDoc-Dataset  28
    # PlantVillage-3_1  39
    class_num = 28  # [可修改]

    net = resnet50(num_classes=class_num)
    net = net.cuda()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "./ResNet50-100epoch-k80-PlantVillage.pth" # [可修改]

    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

    net_weights = net.state_dict() # 读取模型中的权重名称
    pre_weights = torch.load(model_weight_path, map_location=device)
    del_key = []
    for key, _ in pre_weights.items():
        if "fc" in key:
            del_key.append(key)

    for key in del_key:
        del pre_weights[key]

    missing_keys, unexpected_keys = net.load_state_dict(pre_weights, strict=False)
    print("[missing_keys]:", *missing_keys, sep="\n")
    print("[unexpected_keys]:", *unexpected_keys, sep="\n")


    # net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    # in_channel = net.fc.in_features
    # net.fc = nn.Linear(in_channel, class_num)
    # net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 100 # [可修改]
    best_acc = 0.0
    save_path = './resNet50.pth' # [可修改]
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            # train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
            #                                                          epochs,
            #                                                          loss)


        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                # val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                #                                            epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        with open("a.txt", "a") as f:
            f.write('[epoch %d] train_loss: %.3f  val_accuracy: %.3f\n' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')
    with open("a.txt", "a") as f:
        f.write('Finished Training\n')

if __name__ == '__main__':
    main()
