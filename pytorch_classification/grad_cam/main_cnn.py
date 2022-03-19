import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image


def main():
    # parse ------------------------------------------------------------------------------------------------
    img_save_path = './test.jpg'
    img_path = "both.png"
    target_category = 1 #可以是一个列表

    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # ------------------------------------------------------------------------------------------------------

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load model ------------------------------------------------------------------------------------------------
    # model = models.mobilenet_v3_large(pretrained=True)
    # target_layers = [model.features[-1]]

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    model = models.resnet50(num_classes=12, pretrained=False)
    model_weight_path = "./resNet50-PV-health.pth"  # [可修改]
    pre_weights = torch.load(model_weight_path, map_location=device)
    model.load_state_dict(pre_weights, strict=False)
    target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]
    # --------------------------------------------------------------------------------------------------------------

    # load image ------------------------------------------------------------------------------------------------
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)
    # 定义模型
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    # overlays the cam mask on the image as an heatmap
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)

    plt.imshow(visualization)
    plt.savefig(img_save_path)
    # plt.show()



if __name__ == '__main__':
    main()
