from torchstat import stat
from model_v2 import MobileNetV2

model = MobileNetV2()
img_shape = (3, 224, 224)
stat(model, img_shape)
