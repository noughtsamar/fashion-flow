from torchvision import transforms
from dataloaders.data_p2p import VitonDataset
from torch.utils.data import DataLoader
import torch
from models.generator import UnetGenerator,UnetSkipConnectionBlock
from models.discriminator import Discriminator
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision.utils as vutils
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from IPython.display import HTML

data_directory = '../Kapda/dataset'
imageSize = (256,256)
types_folders=['cloth','skeleton','face_segment','agnostic-v3.2']
batch_size=64

transform=transforms.Compose([
    transforms.Resize(imageSize),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

dataset = VitonDataset(data_dir=data_directory, 
                           transform=transform, types_folders=types_folders)
dataloader =  DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator=UnetGenerator(input_nc=12,output_nc=3).to(device)
discriminator=Discriminator(input_nc=15).to(device)

adversarial_loss = nn.BCELoss() 
l1_loss = nn.L1Loss()

def generator_loss(generated_image, target_img, G, real_target):
    gen_loss = adversarial_loss(G, real_target)
    l1_l = l1_loss(generated_image, target_img)
    gen_total_loss = gen_loss + (100 * l1_l)
    return gen_total_loss

def discriminator_loss(output, label):
    disc_loss = adversarial_loss(output, label)
    return disc_loss

G_losses=[]
D_losses=[]
gen_image_list=[]

learning_rate = 2e-4 
G_optimizer = optim.Adam(generator.parameters(), lr = learning_rate, betas=(0.5, 0.999))
D_optimizer = optim.Adam(discriminator.parameters(), lr = learning_rate, betas=(0.5, 0.999))

num_epochs = 30

for epoch in range(num_epochs): 
    dataloader=tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch")
    for i,(input_img, target_img) in enumerate(dataloader):
        
        D_optimizer.zero_grad()
        input_img = input_img.to(device)
        target_img = target_img.to(device)
 
        # ground truth labels real and fake
        real_target = Variable(torch.ones(input_img.size(0), 1, 30, 30).to(device))
        fake_target = Variable(torch.zeros(input_img.size(0), 1, 30, 30).to(device))
         
        # generator forward pass
        generated_image = generator(input_img)
         
        # train discriminator with fake/generated images
        disc_inp_fake = torch.cat((input_img, generated_image), 1)
         
        D_fake = discriminator(disc_inp_fake.detach())
         
        D_fake_loss   =  discriminator_loss(D_fake, fake_target)
         
        # train discriminator with real images
        disc_inp_real = torch.cat((input_img, target_img), 1)
                                 
        D_real = discriminator(disc_inp_real)
        D_real_loss = discriminator_loss(D_real,  real_target)
 
     
         
        # average discriminator loss
        D_total_loss = (D_real_loss + D_fake_loss) / 2
        # compute gradients and run optimizer step
        D_total_loss.backward()
        D_optimizer.step()
         
         
        # Train generator with real labels
        G_optimizer.zero_grad()
        fake_gen = torch.cat((input_img, generated_image), 1)
        G = discriminator(fake_gen)
        G_loss = generator_loss(generated_image, target_img, G, real_target)                                 
        # compute gradients and run optimizer step
        G_loss.backward()
        G_optimizer.step()
        
        dataloader.set_postfix({"Gen Loss": G_loss.item(), "Disc Loss": D_total_loss.item()})
        
        G_losses.append(G_loss.item())
        D_losses.append(D_total_loss.item())
            
        
        if i%100==0:
            with torch.no_grad():
                fake = generator(input_img).detach().cpu()
            gen_image_list.append(vutils.make_grid(fake, padding=2, normalize=True))
    torch.save(generator,'g{}'.format(epoch))
    torch.save(discriminator,'d{}'.format(epoch))

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
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