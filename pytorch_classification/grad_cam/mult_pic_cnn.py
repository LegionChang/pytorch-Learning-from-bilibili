import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from utils import GradCAM, show_cam_on_image
from shutil import copy, rmtree
from torchvision import transforms, datasets

def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 参数 ------------------------------------------------------------------------------------------------
    img_path = "/home/lc/dataset/cam/ResNet-PV-Tomato"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    batch_size = 6
    num_classes = 10    # 你的模型的分类数目
    # ------------------------------------------------------------------------------------------------------

    # 加载图片 ------------------------------------------------------------------------------------------------
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    raw_dataset = datasets.ImageFolder(root=img_path,
                                       transform=data_transform)
    data_loader = torch.utils.data.DataLoader(dataset=raw_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    # ------------------------------------------------------------------------------------------------------

    # 加载你的模型 ------------------------------------------------------------------------------------------------
    model = models.resnet50(num_classes=num_classes, pretrained=False)
    model_weight_path = "./resNet50-PV-health.pth"  # [可修改]
    pre_weights = torch.load(model_weight_path, map_location=device)
    model.load_state_dict(pre_weights, strict=False)
    target_layers = [model.layer4]
    # --------------------------------------------------------------------------------------------------------------

    # 逆归一化 ------------------------------------------------------------------------------------------------
    un_norm = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    # --------------------------------------------------------------------------------------------------------------

    # 建立存放热力图的文件夹----------------------------------------------------------------------------------------
    cam_file_path = img_path + '_cam'
    mk_file(cam_file_path)
    data_list = raw_dataset.class_to_idx
    class_dic = dict((val, key) for key, val in data_list.items())
    for idx in class_dic:
        mk_file(os.path.join(cam_file_path, class_dic[idx]))
    # --------------------------------------------------------------------------------------------------------------

    for step, data in enumerate(data_loader):
        input_tensor = data[0]       # [N, C, H, W]
        target_category = data[1]   # label 列表
        # 热力图生成模型
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category) # [N ,H, W]
        batch_idx = 0
        for img_cam in grayscale_cam:

            img = un_norm(input_tensor[batch_idx]).numpy()  # [C, H, W]
            img = img.transpose(1, 2, 0)   # [H, W, C]
            visualization = show_cam_on_image(img,
                                              img_cam,
                                              use_rgb=True)
            plt.imshow(visualization)
            save_path = os.path.join(cam_file_path, class_dic[target_category.numpy()[batch_idx]])
            pic_no = len(os.listdir(save_path))

            save_pic_whole_path = os.path.join(save_path, '{}.JPG'.format(pic_no))
            plt.savefig(save_pic_whole_path)
            batch_idx += 1
    print("所有图片已经处理结束")
    print("请在一下路径中查看您的热力图：")
    print(cam_file_path)



if __name__ == '__main__':
    main()
