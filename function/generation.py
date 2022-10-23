import torch.nn as nn
import torch
import datetime
import torchvision.utils as vutils

n_channel = 100
n_class = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
        def __init__(self):

            super(Generator, self).__init__()

            # ニューラルネットワークの構造を定義する
            self.layers = nn.ModuleDict({
                'layer0': nn.Sequential(
                    nn.ConvTranspose2d(n_channel+n_class, 512, 3, 1, 0),
                    nn.BatchNorm2d(512),
                    nn.ReLU()
                ), 
                'layer1': nn.Sequential(
                    nn.ConvTranspose2d(512, 256, 3, 2, 0),
                    nn.BatchNorm2d(256),
                    nn.ReLU()
                ),
                'layer2': nn.Sequential(
                    nn.ConvTranspose2d(256, 128, 4, 2, 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU()
                ),
                'layer3': nn.Sequential(
                    nn.ConvTranspose2d(128, 1, 4, 2, 1),
                    nn.Tanh()
                )
            })

        def forward(self, z):

            for layer in self.layers.values(): 
                z = layer(z)
            return z
        
def onehot_encode(label, device):
  
    eye = torch.eye(n_class, device=device)
    return eye[label].view(-1, n_class, 1, 1) 

def concat_image_label(image, label, device):

    B, C, H, W = image.shape
    
    oh_label = onehot_encode(label, device)
    oh_label = oh_label.expand(B, n_class, H, W)
    return torch.cat((image, oh_label), dim=1)

def concat_noise_label(noise, label, device):
  
    oh_label = onehot_encode(label, device)
    return torch.cat((noise, oh_label), dim=1)
        
def generate_image(number, n):
    
    # ノイズとラベル
    label_list = [number]*n
    sample_label = torch.tensor(label_list, dtype=torch.long, device=device)
    sample_noise = torch.randn(n, n_channel, 1, 1, device=device) 
    sample_noise_label = concat_noise_label(sample_noise, sample_label, device) 
    
    # Generatorの定義と画像の生成
    generator = Generator().to('cpu')
    generator.load_state_dict(torch.load('generator.pth', map_location=torch.device('cpu')))
    generator.eval
    y = generator(sample_noise_label)
    
    # 生成画像の保存
    result_dir = "static/generated_images/"
    dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    path_list = []
    for i in range(len(y)):  
        result_img_path = result_dir + dt_now + str(i)+ ".jpg"
        path_list.append(result_img_path)
        vutils.save_image(y[i], result_img_path, normalize=True)
        
    return path_list