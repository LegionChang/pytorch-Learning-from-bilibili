from torchstat import stat
from model import resnet34
import torchvision.models as models

model = models.resnet101()
img_shape = (3, 224, 224)
stat(model, img_shape)
