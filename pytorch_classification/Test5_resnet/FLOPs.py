from torchstat import stat
from model import resnet34
import torchvision.models as models

model = models.vgg13()
img_shape = (3, 224, 224)
stat(model, img_shape)
