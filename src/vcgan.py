import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch 

# Define the generator network
class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.device = args.device
        self.layer = nn.Sequential(
            nn.Linear(args.latent_dim + args.num_classes + 1, args.g_hid[0]),# Added one parameter for roundness
            nn.BatchNorm1d(args.g_hid[0]),
            nn.LeakyReLU(0.3),
            #nn.Sigmoid(),
            #nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(args.g_hid[0], args.g_hid[1]),
            nn.BatchNorm1d(args.g_hid[1]),
            nn.LeakyReLU(0.3),
            #nn.Sigmoid(),
            #nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(args.g_hid[1], args.g_hid[2]),
            nn.BatchNorm1d(args.g_hid[2]),
            nn.LeakyReLU(0.3),
            #nn.Sigmoid(),       
            #nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(args.g_hid[2], args.g_hid[3]),
            nn.BatchNorm1d(args.g_hid[3]),
            nn.LeakyReLU(0.3),
            #nn.Sigmoid(),        
            #nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(args.g_hid[3], args.output_dim),
            #nn.Sigmoid()
            nn.Tanh()
        )
        

        # Initialize weights using nn.init.normal_
        
        # if args.init_type == "normal":
        #     self.apply(self.init_normal)
        # elif args.init_type == "orth":
        #     self.apply(self.init_orth)
        # elif args.init_type == "xavier_uniform":
        #     self.apply(self.init_xavier)
        # else:
        #     raise NotImplementedError(
        #             "{} unknown inital type".format(args.init_type)
        #         )

    def init_normal(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            #nn.init.constant_(m.bias, 0)
    def init_orth(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            #nn.init.constant_(m.bias, 0)
    def init_xavier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            #nn.init.constant_(m.bias, 0)
        
    def forward(self, x, c, r):
        x, c, r = x.to(self.device), c.to(self.device), r.to(self.device)
        x, c, r = x.view(x.size(0), -1), c.view(c.size(0), -1).float(), r.view(c.size(0), -1).float()
        v = torch.cat((x, c, r), 1) # v: [input, label] concatenated vector
        y_ = self.layer(v)
        return y_

# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.device = args.device
        self.layer = nn.Sequential(
            nn.Linear(args.input_dim + args.num_classes + 1, args.d_hid[0]), # Added one parameter for roundness
            nn.LeakyReLU(0.3),
            #nn.Sigmoid(),
            #nn.Dropout(0.1),
            #nn.ReLU(),
            nn.Linear(args.d_hid[0], args.d_hid[1]),
            nn.LeakyReLU(0.3),
            #nn.Sigmoid(),
            #nn.ReLU(),
            nn.Linear(args.d_hid[1], args.d_hid[2]),
            nn.LeakyReLU(0.3),
            #nn.Sigmoid(),
            #nn.ReLU(),
            nn.Linear(args.d_hid[2], args.d_hid[3]),
            nn.LeakyReLU(0.3),
            #nn.Sigmoid(),
            #nn.ReLU(),
            #nn.Dropout(0.1),
            #nn.ReLU(),
            nn.Linear(args.d_hid[3], 1),
            nn.Sigmoid()
        )
        
        # if args.init_type == "normal":
        #     self.apply(self.init_normal)
        # elif args.init_type == "orth":
        #     self.apply(self.init_orth)
        # elif args.init_type == "xavier_uniform":
        #     self.apply(self.init_xavier)
        # else:
        #     raise NotImplementedError(
        #             "{} unknown inital type".format(args.init_type)
        #         )

    def init_normal(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            #nn.init.constant_(m.bias, 0)
    def init_orth(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            #nn.init.constant_(m.bias, 0)
    def init_xavier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            #nn.init.constant_(m.bias, 0)
        
    def forward(self, x, c, r):
        x, c, r = x.to(self.device), c.to(self.device), r.to(self.device)
        x, c, r = x.view(x.size(0), -1), c.view(c.size(0), -1).float(), r.view(c.size(0), -1).float()
        #print(f"X-shape {np.shape(x)}, Y-shape {np.shape(c)}")
        v = torch.cat((x, c, r), 1) # v: [input, label] concatenated vector
        y_ = self.layer(v)
        return y_
    
class DiscriminatorMinus1(nn.Module):
    def __init__(self, args):
        super(DiscriminatorMinus1, self).__init__()
        self.device = args.device
        self.layer = nn.Sequential(
            nn.Linear(args.input_dim + args.num_classes + 1, args.d_hid[0]),# Added one parameter for roundness
            nn.LeakyReLU(0.3),
            #nn.Dropout(0.1),
            #nn.ReLU(),
            #nn.ReLU(),
            nn.Linear(args.d_hid[0], args.d_hid[1]),
            nn.LeakyReLU(0.3),
            #nn.ReLU(),

            nn.Linear(args.d_hid[1], args.d_hid[2]),
            nn.LeakyReLU(0.3),
            #nn.ReLU(),
            nn.Linear(args.d_hid[2], args.d_hid[3]),
            nn.LeakyReLU(0.3)
            #nn.ReLU()
        )
        
        # if args.init_type == "normal":
        #     self.apply(self.init_normal)
        # elif args.init_type == "orth":
        #     self.apply(self.init_orth)
        # elif args.init_type == "xavier_uniform":
        #     self.apply(self.init_xavier)
        # else:
        #     raise NotImplementedError(
        #             "{} unknown inital type".format(args.init_type)
        #         )

    def init_normal(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            #nn.init.constant_(m.bias, 0)
    def init_orth(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            #nn.init.constant_(m.bias, 0)
    def init_xavier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            #nn.init.constant_(m.bias, 0)
        
    def forward(self, x, c, r):
        x, c, r = x.to(self.device), c.to(self.device), r.to(self.device)
        x, c, r = x.view(x.size(0), -1), c.view(c.size(0), -1).float(), r.view(c.size(0), -1).float()

        #print(f"X-shape {np.shape(x)}, Y-shape {np.shape(c)}")
        v = torch.cat((x, c, r), 1) # v: [input, label] concatenated vector
        y_ = self.layer(v)
        return y_