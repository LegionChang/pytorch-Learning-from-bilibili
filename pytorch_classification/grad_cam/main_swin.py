import os
import math
import numpy as np
import torch
from PIL import Image
from shutil import rmtree
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from utils import GradCAM, show_cam_on_image, center_crop_img
from swin_model import swin_tiny_patch4_window7_224


class ResizeTransform:
    def __init__(self, im_h: int, im_w: int):
        self.height = self.feature_size(im_h)
        self.width = self.feature_size(im_w)

    @staticmethod
    def feature_size(s):
        s = math.ceil(s / 4)  # PatchEmbed
        s = math.ceil(s / 2)  # PatchMerging1
        s = math.ceil(s / 2)  # PatchMerging2
        s = math.ceil(s / 2)  # PatchMerging3
        return s

    def __call__(self, x):
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        result = x.reshape(x.size(0),
                           self.height,
                           self.width,
                           x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)

        return result

def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)

def main():
    # 注意输入的图片必须是32的整数倍
    # 否则由于padding的原因会出现注意力飘逸的问题

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_size = 256
    assert img_size % 32 == 0
    # 参数 ------------------------------------------------------------------------------------------------
    img_path = "/home/lc/dataset/cam/Swin-PV-Tomato"
    # img_path = "./mult_pic_test"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    batch_size = 32
    num_classes = 12    # 你的模型的分类数目
    weights_path = "./swin-tiny-PV-health.pth"  # 模型权重文件路径
    # ------------------------------------------------------------------------------------------------------

    # 加载图片 ------------------------------------------------------------------------------------------------
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    raw_dataset = datasets.ImageFolder(root=img_path,
                                       transform=data_transform)
    data_loader = torch.utils.data.DataLoader(dataset=raw_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    print("总共有{}张图片需要处理".format(len(raw_dataset)))
    # ------------------------------------------------------------------------------------------------------

    # 加载你的模型 ------------------------------------------------------------------------------------------------
    model = swin_tiny_patch4_window7_224(num_classes=num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    target_layers = [model.norm]
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

    print("开始处理热力图")
    batch_num = 1  # 用于记录进度
    for step, data in enumerate(data_loader):
        input_tensor = data[0]  # [N, C, H, W]
        target_category = data[1]  # label 列表
        # 热力图生成模型
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True,
                      reshape_transform=ResizeTransform(im_h=img_size, im_w=img_size))
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)  # [N ,H, W]

        batch_idx = 0
        for img_cam in grayscale_cam:
            img = un_norm(input_tensor[batch_idx]).numpy()  # [C, H, W]
            img = img.transpose(1, 2, 0)  # [H, W, C]
            visualization = show_cam_on_image(img,
                                              img_cam,
                                              use_rgb=True)
            plt.imshow(visualization)
            save_path = os.path.join(cam_file_path, class_dic[target_category.numpy()[batch_idx]])
            pic_no = len(os.listdir(save_path))

            save_pic_whole_path = os.path.join(save_path, '{}.JPG'.format(pic_no))
            plt.savefig(save_pic_whole_path)
            batch_idx += 1
        current_progress = batch_num * batch_size / len(raw_dataset) * 100
        print("当前已处理{:.2f}%".format(current_progress))
        batch_num += 1
    print("所有图片已经处理结束")
    print("请在一下路径中查看您的热力图：")
    print(cam_file_path + '\n')



if __name__ == '__main__':
    main()
