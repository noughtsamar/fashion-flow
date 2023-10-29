from torchvision import transforms
from dataloaders.data_without_disc import VitonDataset
from torch.utils.data import DataLoader
from models.generator import UnetGenerator,UnetSkipConnectionBlock
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from IPython.display import HTML

data_directory = 'data'
imageSize = (256,256)
types_folders=['cloth','skeleton','face_segment','agnostic-v3.2','image']
batch_size=64

transform=transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.Normalize(0.5,0.5)
    ])

dataset = VitonDataset(data_dir=data_directory, 
                           transform=transform, types_folders=types_folders,imageSize=imageSize)
dataloader =  DataLoader(dataset, batch_size=batch_size, shuffle=True ,num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator=UnetGenerator(input_nc=12,output_nc=3).to(device)

l1_loss = nn.L1Loss()

def generator_loss(generated_image, target_img):
    return (l1_loss(generated_image, target_img)*100)

G_losses=[]
gen_image_list=[]

learning_rate = 2e-4 
G_optimizer = optim.Adam(generator.parameters(), lr = learning_rate, betas=(0.5, 0.999))

num_epochs = 30

for epoch in range(num_epochs): 
    dataloader=tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch")
    for i,(input_img, target_img) in enumerate(dataloader):
        
        input_img = input_img.to(device)
        target_img = target_img.to(device)
 
        # generator forward pass
        generated_image = generator(input_img)
         
        # Train generator with real labels
        G_optimizer.zero_grad()

        G_loss = generator_loss(generated_image, target_img)                                 
        # compute gradients and run optimizer step
        G_loss.backward()
        G_optimizer.step()
        
        dataloader.set_postfix({"Gen Loss": G_loss.item()})
        
        G_losses.append(G_loss.item())
    with torch.no_grad():
        fake = generator(input_img).detach().cpu()
    gen_image_list.append(vutils.make_grid(fake, padding=2, normalize=True))
    torch.save(generator,'g{}'.format(epoch))

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('training_loss_plot.png')
plt.show()

fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in gen_image_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())
ani.save("training_progress.mp4", writer="ffmpeg", fps=2)
