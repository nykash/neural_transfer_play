import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

###
# inspiration from https://www.youtube.com/watch?v=imX4kSKDY7s
###


model = models.vgg19(True).features
print(model)

class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = ["0", "5", "10", "19", "28"]
        self.model = models.vgg19(True).features[:29]

    def forward(self, x):
        features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if(str(layer_num) in self.features):
                features.append(x)

        return features

def load_image(path):
    img = Image.open(path)
    image = loader(img).unsqueeze(0)
    return image.to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_size = 356
loader = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor()
])

og_img = load_image("cherry_blossom.jpg")
style_img = load_image("anime_cherry_blossomjpg.jpg")

gen = og_img.clone().requires_grad_(True)
tot_steps = 6000
lr=0.001
alpha = 1
beta = 0.01

model = VGG().to(device).eval()

optimizer = optim.Adam([gen], lr=lr)

for step in range(tot_steps):
    generated_features = model(gen)
    og_img_feat = model(og_img)
    style_features = model(style_img)

    style_loss = 0
    original_loss = 0

    for gen_feature, orig_feature, style_feature in zip(generated_features, og_img_feat, style_features):
        n, c, h, w = gen_feature.shape
        original_loss += torch.mean((gen_feature-orig_feature)**2)

        G = gen_feature.view(c, h*w).mm(gen_feature.view(c, h*w).t())
        A = style_feature.view(c, h * w).mm(style_feature.view(c, h * w).t())

        style_loss += torch.mean((G-A)**2)

    total_loss = alpha * original_loss + beta * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if(step % 200 == 0):
        print(total_loss)
        save_image(gen, f"generated_{step}.png")

