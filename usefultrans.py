from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open('hymenoptera_data/train/ants/0013035.jpg')


trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image('img_tensor', img_tensor)

# normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.485, 0.456, 0.406],[0.485, 0.456, 0.406])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
# 感觉可以理解为图像变换
writer.add_image('img_norm', img_norm)

# resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
writer.add_image('img_resize', img_resize, 0)
print(img_resize.size)

# compose
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image('img_resize', img_resize_2, 1)

# randomCrop
trans_random = transforms.RandomCrop(128)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image('img_crop', img_crop, i)
writer.close()
