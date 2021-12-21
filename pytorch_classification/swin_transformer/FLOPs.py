from torchstat import stat

from model import swin_tiny_patch4_window7_224

model = swin_tiny_patch4_window7_224()
img_shape = (3, 224, 224)
stat(model, img_shape)
