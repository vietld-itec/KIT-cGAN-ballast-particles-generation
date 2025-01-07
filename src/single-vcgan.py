from datasets import generate_real_data, generate_noise, generate_random_data, random_data, random_batch_data
from vcgan import Generator, Discriminator, DiscriminatorMinus1  # Import Generator and Discriminator for CGAN model
from myplot import plot_polygons, plot_save_polygons

import dill
import torch # Import Generator
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import argparse
import numpy as np
import csv
import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,roc_auc_score, mean_squared_error, r2_score

from scipy.stats import energy_distance

class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        label = torch.tensor(sample[0], dtype=int)  # Assuming the label is in the first column
        roundness = torch.tensor(sample[1], dtype=torch.float32)
        data_points = torch.tensor(sample[2:], dtype=torch.float32)  # Assuming data points start from the second column
        return label,roundness, data_points

class CustomArgs:
    def __init__(self, path_file='csvdat.csv',classes = ['angular', 'rounded', 'subangular', 'subrounded'], sample_len=64, sample_size=2, sample_startid=0,
                 record_file='loss.csv', g_hid=None, d_hid=None, num_classes = 4, init_type="xavier_uniform", kernel = "rbf",
                 g_lr=0.0002, d_lr=0.0002, beta1=0.0, beta2=0.9, input_dim=64, output_dim=64,
                 latent_dim=64, n_critic=1, num_epochs=10, min_epochs = 10, num_patience=10, row=1, batch_size=2):
        self.path_file = path_file
        self.classes = classes
        self.sample_len = sample_len
        self.sample_size = sample_size
        self.sample_startid = sample_startid
        self.record_file = record_file
        self.g_hid = g_hid if g_hid is not None else [128, 256, 128]  # Provide a default value or replace with your specific values
        self.d_hid = d_hid if d_hid is not None else [128, 256, 128, 64]  # Provide a default value or replace with your specific values
        self.num_classes = num_classes
        self.init_type = init_type
        self.kernel = kernel
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.min_epochs = min_epochs
        self.num_epochs = num_epochs
        self.patience = num_patience
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.row = row
        self.col = sample_size
        self.batch_size = batch_size


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0, save_path='checkpoint.pt'):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, gen_loss, disc_loss, model):
        diff_loss = gen_loss #abs(gen_loss + disc_loss)
        print(f"count:{self.counter},RMSE_value:{diff_loss},Min. RMSE:{self.best_loss}")
        if self.best_loss is None:
            self.best_loss = diff_loss
            self.save_checkpoint(model)
        elif diff_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
        elif abs(diff_loss - self.best_loss) <= 1e-3:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            self.best_loss = diff_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        with open(self.save_path, 'wb') as f:
            torch.save(model, f)
    def save_epoch(self,save_path, model):
        with open(save_path, 'wb') as f:
            torch.save(model, f)
    @staticmethod
    def load_checkpoint(load_path):
        with open(load_path, 'rb') as f:
            model.load_state_dict(torch.load(f))
        return model
def tensor_normalize(tensor_matrix):
    dimval = np.shape(tensor_matrix)
    nrows = dimval[0]
    ncols = dimval[1]
    normalized_matrix = tensor_matrix.clone().detach()
    for r in range(nrows):
        sumrow = 0
        for c in range(ncols):
            sumrow += tensor_matrix[r,c]
        for c in range(ncols):
            val = tensor_matrix[r,c]/sumrow
            normalized_matrix[r,c] = val
    return normalized_matrix


def avg_roc_calc(args,discriminator, fake_datas, y_target):
    # Get ;ist of classes
    classes = args.classes
    y_proba = torch.zeros(args.batch_size, args.num_classes, dtype=float) # The label for fake data is 0

    for k in range(len(classes)):
        desired_label = torch.zeros(args.batch_size, 4, dtype=int) # The label for fake data is 0
        desired_label[:,k] = 1
        z_outputs = discriminator(fake_datas,desired_label) # Generate fake data
        y_proba[:,k] = z_outputs[:,0].clone().detach()
    y_proba = tensor_normalize(y_proba)
    #Calculate the partially roc value
    roc_auc_ovr = {}
    for i in range(len(classes)):
        # Gets the class
        c = classes[i]
        #print(f"DEBUG: y:{y}\nreal:{y_target[:, i]}\n predicted:{y_proba[:, i]}")
        df_aux = pd.DataFrame({'class': y_target[:, i].cpu().numpy(),'prob': y_proba[:, i].cpu().numpy() })
        df_aux = df_aux.reset_index(drop = True)

        roc_auc_ovr[c] = roc_auc_score(df_aux['class'], df_aux['prob'])
    #Calculate avg roc score
    avg_roc_auc = 0
    i = 0
    for k in roc_auc_ovr:
        avg_roc_auc += roc_auc_ovr[k]
        i += 1
    return avg_roc_auc/i
def calculate_tpr_fpr(y_real, y_pred):
    '''
    Calculates the True Positive Rate (tpr) and the True Negative Rate (fpr) based on real and predicted observations
    
    Args:
        y_real: The list or series with the real classes
        y_pred: The list or series with the predicted classes
        
    Returns:
        tpr: The True Positive Rate of the classifier
        fpr: The False Positive Rate of the classifier
    '''
    
    # Calculates the confusion matrix and recover each element
    cm = confusion_matrix(y_real, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    
    # Calculates tpr and fpr
    tpr =  TP/(TP + FN) # sensitivity - true positive rate
    fpr = 1 - TN/(TN+FP) # 1-specificity - false positive rate
    
    return tpr, fpr

def get_all_roc_coordinates(y_real, y_proba):
    '''
    Calculates all the ROC Curve coordinates (tpr and fpr) by considering each point as a threshold for the predicion of the class.
    
    Args:
        y_real: The list or series with the real classes.
        y_proba: The array with the probabilities for each class, obtained by using the `.predict_proba()` method.
        
    Returns:
        tpr_list: The list of TPRs representing each threshold.
        fpr_list: The list of FPRs representing each threshold.
    '''
    tpr_list = [0]
    fpr_list = [0]
    for i in range(len(y_proba)):
        threshold = y_proba[i]
        y_pred = y_proba >= threshold
        tpr, fpr = calculate_tpr_fpr(y_real, y_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list
def plot_roc_curve(tpr, fpr, scatter = True, ax = None):
    '''
    Plots the ROC Curve by using the list of coordinates (tpr and fpr).
    
    Args:
        tpr: The list of TPRs representing each coordinate.
        fpr: The list of FPRs representing each coordinate.
        scatter: When True, the points used on the calculation will be plotted with the line (default = True).
    '''
    if ax == None:
        plt.figure(figsize = (5, 5))
        ax = plt.axes()
    
    if scatter:
        sns.scatterplot(x = fpr, y = tpr, ax = ax)
    sns.lineplot(x = fpr, y = tpr, ax = ax)
    sns.lineplot(x = [0, 1], y = [0, 1], color = 'green', ax = ax)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("False Positive Rate", fontname="Times New Roman", fontsize=14, fontweight = 'bold')
    plt.ylabel("True Positive Rate", fontname="Times New Roman", fontsize=14, fontweight = 'bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    for tick in ax.get_xticklabels():
        tick.set_fontname("Times New Roman")
    for tick in ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
def to_onehot(x, num_classes=4):
    # Convert original vector to one-hot encoded tensor
    #print(f" x:{np.shape(x)}")
    #print(f"{x}")
    onehot_tensor = torch.zeros(len(x), num_classes)
    #print(f" x:{np.shape(onehot_tensor)}")
    #onehot_tensor.scatter_(1, x.unsqueeze(1), 1)
    #x = x.to(torch.int64)
    onehot_tensor.scatter_(1, x, 1)
    #onehot_tensor.scatter_(1, x.unsqueeze(1), 1)
    return onehot_tensor

def to_onehot_back(x, num_classes, device):
    assert isinstance(x, int) or isinstance(x, (torch.LongTensor, torch.cuda.LongTensor))
    if isinstance(x, int):
        c = torch.zeros(1, num_classes, device=device).long()
        c[0][x] = 1
    else:
        x = x.cpu()
        c = torch.zeros(x.size(0), num_classes, device=device).long()
        c.scatter_(1, x.unsqueeze(1), 1)  # dim, index, src value
    return c
    
def rmse_percentage(measured_data, predicted_data):
    # Convert Torch tensors to NumPy arrays
    # Detach tensors from the computation graph and move to CPU
    measured_data_np = measured_data.detach().cpu().numpy()
    predicted_data_np = predicted_data.detach().cpu().numpy()
    
    # Calculate RMSE
    array_rmse = []
    #print(f"{measured_data_np.shape[0]}")
    for i in range(measured_data_np.shape[0]):
        rmse = mean_squared_error(measured_data_np[i], predicted_data_np[i], squared=False)
        array_rmse.append(rmse)
    # Compute average
    average_rmse = np.mean(array_rmse)
    return average_rmse*100

def r2score_percentage(measured_data, predicted_data):
    # Convert Torch tensors to NumPy arrays
    # Detach tensors from the computation graph and move to CPU
    measured_data_np = measured_data.detach().cpu().numpy()
    predicted_data_np = predicted_data.detach().cpu().numpy()
    # Calculate R2 score for each row
    r2_scores = []
    #print(f"{measured_data_np.shape[0]}")
    for i in range(measured_data_np.shape[0]):
        r2 = r2_score(measured_data_np[i], predicted_data_np[i])
        r2_scores.append(r2)

    # Compute average R2 score
    average_r2_score = np.mean(r2_scores)
    return average_r2_score*100

def combinematrix(real_label, pred_label):
    with torch.no_grad():
        real_label = real_label.cpu().numpy()
        pred_label = pred_label.cpu().numpy()
    predicted_matrix = [[pred_label[i][0] * real_label[i][j] if real_label[i][j] != 0 else 0 for j in range(len(real_label[i]))] for i in range(len(real_label))]
    return np.array(real_label), np.array(predicted_matrix)

def save_model(_save_path, _model):
        with open(_save_path, 'wb') as f:
            torch.save(_model, f)
# Discriminator objective function
def discriminator_loss(real_output, fake_output):
    # Calculate the loss for real samples
    real_loss = F.binary_cross_entropy_with_logits(real_output, torch.ones_like(real_output))
    # Calculate the loss for fake samples
    fake_loss = F.binary_cross_entropy_with_logits(fake_output, torch.zeros_like(fake_output))
    # Total loss is the sum of losses for real and fake samples
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(D_minus1, y_label, x_real, z_real, z_prime, G, alpha):
    """
    Compute the generator loss based on the provided equation.

    Args:
    - D_minus1 (torch.nn.Module): Discriminator excluding the last layer
    - x_real (torch.Tensor): Real samples from the true distribution Pr
    - z_real (torch.Tensor): Real samples from the noise distribution Pz
    - G (torch.nn.Module): Generator model
    - z_prime (torch.Tensor): Noise samples for computing z_prime

    Returns:
    - torch.Tensor: Generator loss
    """
    # Compute D_minus1(x) for real samples
    D_minus1_x_real = D_minus1(x_real, y_label)

    # Compute D_minus1(G(z)) for generated samples
    x_fake = G(z_real, y_label)
    D_minus1_G_z_real = D_minus1(x_fake, y_label)

    # Compute D_minus1(x') for i.i.d. copies of x_real
    x_prime = x_real.clone().detach()
    D_minus1_x_prime = D_minus1(x_prime, y_label)

    # Compute D_minus1(G(z')) for i.i.d. copies of z_real
    z_prime_samples = z_prime.clone().detach()
    x_fake_prime = G(z_prime_samples, y_label)
    D_minus1_G_z_prime = D_minus1(x_fake_prime, y_label)

    # Compute the loss terms
    loss_term1 = torch.norm(D_minus1_x_real - D_minus1_G_z_real,p=2,  dim=1)**alpha
    loss_term2 = torch.norm(D_minus1_x_real - D_minus1_x_prime,p=2,  dim=1)**alpha
    loss_term3 = torch.norm(D_minus1_G_z_real - D_minus1_G_z_prime,p=2,  dim=1)**alpha

    # Compute the final generator loss
    generator_loss = 2 * torch.mean(loss_term1) - torch.mean(loss_term2) - torch.mean(loss_term3)

    return generator_loss
def MMD(x, y, args):
    """Emprical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(args.device),
                  torch.zeros(xx.shape).to(args.device),
                  torch.zeros(xx.shape).to(args.device))

    if args.kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if args.kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)



    return torch.mean(XX + YY - 2. * XY)
def average_energy_distance(measured_data, predicted_data):
    # Convert Torch tensors to NumPy arrays
    # Detach tensors from the computation graph and move to CPU
    measured_data_np = measured_data.detach().cpu().numpy()
    predicted_data_np = predicted_data.detach().cpu().numpy()
    
    # Calculate RMSE
    array_rmse = []
    #print(f"{measured_data_np.shape[0]}")
    for i in range(measured_data_np.shape[0]):
        rmse = energy_distance(measured_data_np[i], predicted_data_np[i])
        array_rmse.append(rmse)
    # Compute average
    average_rmse = np.mean(array_rmse)
    return average_rmse
def run_model(args, train_loader, filename):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # The device to use
    #print(f"batch_size: {batch_size}")
    print(f"batch_size:{args.batch_size}")
    # Path to the existing CSV file or create a new one if it doesn't exist
    csv_file_path = f"{filename}.csv"
    
    img_path = filename
    # Check if the folder already exists or not
    if not os.path.exists(img_path):
        # Create the folder
        os.makedirs(img_path)
        print(f"Folder '{img_path}' created successfully.")
    else:
        print(f"Folder '{img_path}' already exists.")
    
    final_d_loss = 0
    final_g_loss = 0
    final_epoch = 0
    
    # Create the generator and discriminator networks
    generator = Generator(args).to(args.device)
    discriminator = Discriminator(args).to(args.device)
    discriminatorminus1 = DiscriminatorMinus1(args).to(args.device)

    # Define the loss function and the optimizers
    bce =  torch.nn.MSELoss() # The binary cross entropy loss

    optimizer_G = optim.Adam(generator.parameters(), lr=args.g_lr, betas=(args.beta1, args.beta2)) # The optimizer for the generator
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(args.beta1, args.beta2)) # The optimizer for the discriminator
    optimizer_Dminus1 = optim.Adam(discriminatorminus1.parameters(), lr=args.d_lr, betas=(args.beta1, args.beta2)) # The optimizer for the discriminator 
    
    #optimizer_G = optim.SGD(generator.parameters(), lr=args.d_lr)
    #optimizer_D = optim.SGD(discriminator.parameters(), lr=args.d_lr)
    #optimizer_Dminus1 = optim.SGD(discriminatorminus1.parameters(), lr=args.d_lr) # The optimizer for the discriminator 
    
    # Define early stopping parameters
    early_stopping_patience = args.patience
    early_stopping_counter = 0
    best_generator_loss = float('inf')

    # Create an instance of the early stopping class
    early_stopping = EarlyStopping(tolerance=early_stopping_patience, min_delta=0, save_path=f'{filename}_best.pt')
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # The device to use
    # Train the GANs
    #batch_size = args.batch_size
    D_labels = torch.ones([args.batch_size, 1]).to(args.device) # Discriminator Label to real
    D_fakes = torch.zeros([args.batch_size, 1]).to(args.device) # Discriminator Label to fake
    
    # Open the CSV file for writing or appending
    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = ['Epoch', 'Generator Loss', 'Discriminator Loss', 'RMSE', 'Ds']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # If the file is empty, write the header
        if csvfile.tell() == 0:
            writer.writeheader()
        for epoch in range(args.num_epochs):
            # Initialize the epoch losses
            epoch_rmse = 0
            epoch_Ds = 0
            epoch_g_loss = 0
            epoch_d_loss = 0
            len_batch = 0
            final_d_loss = 0
            final_g_loss = 0
            final_epoch = 0
            final_Ds_loss = 0
            #args.batch_size = batch_size
            for batch_idx, (labels, roundness, data_points) in enumerate(train_loader):
                len_batch += 1
                
                r_real = roundness.view(args.batch_size, 1)
                # Training Discriminator
                real_datas = random_batch_data(data_points, args.batch_size)
                x_real = real_datas.to(args.device)
                y_real_labels = labels.view(args.batch_size, 1)
                #print(y)
                y_real_labels = to_onehot(y_real_labels).to(args.device)
                #print(f" x:{np.shape(x)},y:{np.shape(y)},v:{np.shape(v)}")
                #y = to_onehot(y).to(args.device)
                x_outputs = discriminator(x_real, y_real_labels,r_real)
                D_x_loss = bce(x_outputs, D_labels) # Calculate the loss for real data
                
                z_noise_real = generate_noise(args) # Generate noise
                fake_datas = generator(z_noise_real, y_real_labels,r_real)
                z_outputs = discriminator(fake_datas,y_real_labels,r_real) # Generate fake data
                D_z_loss = bce(z_outputs, D_fakes)
                d_loss = 0.5*(D_x_loss + D_z_loss) # Calculate the total loss for the discriminator
                #d_loss = MMD(real_datas, fake_datas, args)
                #d_loss = discriminator_loss(real_datas, fake_datas)

                #discriminator.zero_grad()
                optimizer_D.zero_grad()
                d_loss.backward() # Backpropagate the loss
                optimizer_D.step() # Update the parameters
                
                if epoch % args.n_critic == 0:
                    
                    # Training Generator
                    z_noise_prime = generate_noise(args) # Generate noise
                    fake_datas = generator(z_noise_prime, y_real_labels,r_real)
                    z_outputs = discriminator(fake_datas, y_real_labels,r_real)
                    g_loss = bce(z_outputs, D_labels)
                    #g_loss = MMD(real_datas, fake_datas, args)
                    #g_loss = generator_loss(discriminatorminus1, y_real_labels, x_real, z_noise_real, z_noise_prime, generator, alpha=0.75)
                    #print(f"******DEBUG:z_output: {torch.mean(z_outputs)}")
                    #generator.zero_grad() 
                    optimizer_G.zero_grad()
                    optimizer_Dminus1.zero_grad()
                    g_loss.backward() # Backpropagate the loss
                    optimizer_G.step() # Update the parameters
                    optimizer_Dminus1.step()
                    # Calculate the ROC score
                    #avg_roc = avg_roc_calc(args,discriminator, fake_datas, y)

                    # Calculate loss
                    epoch_g_loss += g_loss.item()
                    epoch_d_loss += d_loss.item()
                    epoch_rmse += rmse_percentage(real_datas, fake_datas)
                    epoch_Ds += average_energy_distance(real_datas, fake_datas) #(1- avg_roc) #r2score_percentage(real_datas, fake_datas)
                    if epoch % 1000 == 0:#with torch.no_grad():
                        with torch.no_grad():
                            generator.eval()
                            fake_datas = generator(z_noise_real, y_real_labels,r_real)
                            fake_datas = fake_datas.cpu().numpy()
                            chrttitle = f"Epoch:{epoch}, D_loss: {epoch_d_loss:.4f}, G_loss: {epoch_g_loss:.4f}, Ds: {epoch_Ds:.4f}"
                            plot_save_polygons(fake_datas, chrttitle, args.row, args.col, save_path=f"{img_path}/fake_epoch{epoch}_{batch_idx}.png", dpi=500)
                            generator.train()
                            real_datas = real_datas.cpu().numpy()
                            chrttitle = f"Epoch:{epoch}, D_loss: {epoch_d_loss:.4f}, G_loss: {epoch_g_loss:.4f}, Ds: {epoch_Ds:.4f}"
                            plot_save_polygons(real_datas, chrttitle, args.row, args.col, save_path=f"{img_path}/real_epoch{epoch}_{batch_idx}.png", dpi=500)
            # Compute the average epoch losses
            epoch_g_loss /= len_batch
            epoch_d_loss /= len_batch
            epoch_rmse /= len_batch
            epoch_Ds /= len_batch
            # Print the losses and plot the generated polygons
            print(f'Epoch {epoch}, D_loss: {epoch_d_loss:.4f}, G_loss: {epoch_g_loss:.4f}, RMSE: {epoch_rmse:.4f}, Ds: {epoch_Ds:.4f}')
            writer.writerow({'Epoch': epoch, 'Generator Loss': epoch_g_loss, 'Discriminator Loss': epoch_d_loss, 'RMSE': epoch_rmse, 'Ds': epoch_Ds})
            # Check for early stopping
            # Check early stopping
            if epoch >= args.min_epochs:
                early_stopping(epoch_Ds, epoch_d_loss, 
                {
                    "args": args,
                    "latent_dim": args.latent_dim,
                    "input_dim": args.input_dim,
                    "output_dim": args.output_dim,
                    "init_type": args.init_type,
                    "g_hid": args.g_hid,
                    "d_hid": args.d_hid,
                    "rmse": epoch_rmse,
                    "Ds_loss": epoch_Ds,
                    "g_loss": epoch_g_loss,
                    "d_loss": epoch_d_loss,
                    "epoch": epoch,
                    "gen_state_dict": generator.state_dict(),
                    "dis_state_dict": discriminator.state_dict()
                })
                #viet = False
                #if epoch == 5000:
                #    viet = True
                #     y_target, y_proba = combinematrix(y, z_outputs)
                best_generator = generator
                best_discriminator = discriminator
                final_g_loss = epoch_g_loss
                final_d_loss = epoch_d_loss
                final_epoch = epoch
                final_Ds_loss = epoch_Ds
                # #print(f"DEBUG: real:{y_target}\n predicted:{y_proba}")
                # if epoch >= args.min_epochs:
                #     early_stopping.save_epoch( f'{filename}_epoch{epoch}.pt',
                #     {
                #         "args": args,
                #         "latent_dim": args.latent_dim,
                #         "input_dim": args.input_dim,
                #         "output_dim": args.output_dim,
                #         "init_type": args.init_type,
                #         "g_hid": args.g_hid,
                #         "d_hid": args.d_hid,
                #         "rmse": epoch_rmse,
                #         "roc_loss": epoch_roc_loss,
                #         "g_loss": epoch_g_loss,
                #         "d_loss": epoch_d_loss,
                #         "epoch": epoch,
                #         "gen_state_dict": generator.state_dict(),
                #         "dis_state_dict": discriminator.state_dict()
                #     })                          
                if early_stopping.early_stop: # or (epoch == args.min_epochs and epoch_rmse > 50.0):
                #viet = False
                #if epoch == 10000:
                #    viet = True
                #if viet==True:
                    # Load the best model
                    checkpoint = torch.load(f'{filename}_best.pt')

                    args = checkpoint["args"]
                    
                    final_g_loss = checkpoint["g_loss"]
                    final_d_loss = checkpoint["d_loss"]
                    final_epoch = checkpoint["epoch"]
                    final_Ds_loss = checkpoint["Ds_loss"]
                    
                    best_generator = Generator(args).to(args.device)
                    best_discriminator = Discriminator(args).to(args.device)
                    
                    best_generator.load_state_dict(checkpoint["gen_state_dict"])
                    best_discriminator.load_state_dict(checkpoint["dis_state_dict"])
                    
                    print(f"Early stopping at epoch {epoch}")
                    #if viet==True:
                    break
            #if epoch % 10 == 0:#with torch.no_grad():
            #    #args.batch_size = 12
            #    with torch.no_grad():
            #        desired_label = torch.ones(args.batch_size, 1, dtype=int) # The label for fake data is 0
            #        desired_label = to_onehot(desired_label).to(device)
            #        noise = generate_noise(args) # Generate noise for 16 samples
            #        fake_data = generator(noise,desired_label) # Generate fake data
            #        fake_data = fake_data.cpu().numpy() # Convert to numpy array
            #        #fake_data = fake_data.tolist() #fake_data.cpu().numpy()# torch.tensor(fake_data, device=device)#cpu().fake_data.numpy().argmax()#np.array(fake_data.detach().cpu().numpy())
            #        #print(f"fake_data: {np.shape(fake_data )}")
            #        #plot_polygons(fake_data, f"Epoch:{epoch}", args.row, args.col) # Plot the polygons(fake_data, epoch) # Plot the polygons
            #        plot_save_polygons(fake_data, f"Epoch:{epoch}, D_loss: {epoch_d_loss:.4f}, G_loss: {epoch_g_loss:.4f}, Fitness: {100-epoch_rmse:.2f}%"
            #        , 2, 5, save_path=f"{img_path}/epoch{epoch}.png", dpi=500)
                    
    return generator, discriminator, best_generator, best_discriminator, final_g_loss, final_d_loss, final_epoch, final_Ds_loss


# Input range of NAS
#g_hid = np.array([ [32, 64, 128],[64, 128, 256], [64, 128, 64], [64, 64, 64]])
#d_hid = np.array([ [128, 256, 128, 64], [256, 512, 256, 128], [128, 128, 128, 64]])

# g_hid = np.array([[364, 728, 1092, 1456],
#  [364, 728, 1456, 1092],
#  [364, 1092, 1456, 728],
#  [364, 1456, 1092, 728],
#  [728, 1092, 1456, 364],
#  [728, 1456, 1092, 364],
#  [1092, 1456, 728, 364],
#  [1456, 1092, 728, 364]])

# d_hid = np.array([[[364, 728, 1092, 1456],
#  [364, 728, 1456, 1092],
#  [364, 1092, 1456, 728],
#  [364, 1456, 1092, 728],
#  [728, 1092, 1456, 364],
#  [728, 1456, 1092, 364],
#  [1092, 1456, 728, 364],
#  [1456, 1092, 728, 364]]])

g_hid = np.array([[128, 256, 512, 1024],       
 [128, 256, 1024, 512],
 [128, 512, 256, 1024],
 [128, 1024, 512, 256],
 [256, 512, 1024, 128],
 [256, 1024, 512, 128],
 [512, 1024, 256, 128],
 [1024, 512, 256, 128]])
d_hid = np.array([[128, 256, 512, 1024],
 [128, 256, 1024, 512],
 [128, 512, 256, 1024],
 [128, 1024, 512, 256],
 [256, 512, 1024, 128],
 [256, 1024, 512, 128],
 [512, 1024, 256, 128],
 [1024, 512, 256, 128]])

lr_range = np.array([1e-5, 1e-4, 1e-3, 1e-2, 0.1])


# g_hid = np.array([[128, 256, 512, 1024]])
# d_hid = np.array([
#  [256, 512, 1024, 128]])

#lr_range = np.array([1e-5, 2*1e-5, 4*1e-5, 6*1e-5, 8*1e-5, 1e-4])
#lr_range = np.array([1e-4, 1e-3, 1e-2, 0.1])

# Specify the path of the folder you want to create


#args = argparse.ArgumentParser()

#SELECTED model

# Selected model
g_index = 3
d_index = 6
lr_index = 0


g_hid_value = g_hid[g_index]
d_hid_value = d_hid[d_index]
lr_value = lr_range[lr_index]

args = CustomArgs()

nsamples = 360

folder_path = f'selected-vcgan-{nsamples}'

args.path_file = f'csv/csvdat{nsamples}plus.csv'
args.sample_len = nsamples  # number N of points where a curve is sampled
args.sample_size = 40  # number of curves in the training set
args.sample_startid = 0
args.classes = ['Angular', 'Rounded', 'Subangular', 'Subrounded']
args.record_file = 'loss.csv'

args.g_hid = g_hid[0]
args.d_hid = d_hid[0]
args.num_classes = 4

args.init_type = "xavier_uniform" #"xavier_uniform" #"normal" #"xavier_uniform" # normal , orth
args.kernel = "rbf"
args.g_lr = 0.0002
args.d_lr = 0.0002
args.beta1 = 0.5
args.beta2 = 0.999
args.input_dim = nsamples
args.output_dim = nsamples
args.latent_dim = 100
args.n_critic = 1

args.num_epochs = 500000
args.patience = 50000
args.min_epochs = 50000

args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # The device to use

# Plot polygon
args.row = 5
args.col = 8 #args.sample_size

args.batch_size = 40

# Example of creating an instance with specific values
# save_args = CustomArgs(path_file=args.path_file, sample_len=args.sample_len, sample_size=args.sample_size, sample_startid=args.sample_startid,
#                  record_file='loss.csv', g_hid=g_hid[0], d_hid=d_hid[0], init_type=args.init_type,
#                  g_lr=0.0002, d_lr=0.0002, beta1=0.0, beta2=0.9, input_dim=64, output_dim=64, latent_dim=64)

# Check if the folder already exists or not
if not os.path.exists(folder_path):
    # Create the folder
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created successfully.")
else:
    print(f"Folder '{folder_path}' already exists.")
    
main_csv_file = f"{folder_path}/combination.csv"



batch_size = args.batch_size

# Create custom dataset
dataset = CustomDataset(args.path_file)

print(f"Dataset : len{len(dataset)}")

# Create data loader
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# Initial data


#real_data = generate_real_data(args) # Generate real data

#real_data = real_data.cpu().numpy()
#print(f"shape of data: {np.shape(real_data)}")

#plot_polygons(real_data, "Initial data", args.row, args.col) # Plot the polygons
#plot_save_polygons(real_data, f"Initial data", args.row, args.col, save_path=f"{folder_path}/dataset.png", dpi=500)


# Open the CSV file for writing or appending
with open(main_csv_file, 'a', newline='') as csvfile:
    fieldnames = ['id', 'g_index', 'd_index', 'lr','g_loss', 'd_loss', 'stop_epoch', 'roc']
    main_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # If the file is empty, write the header
    if csvfile.tell() == 0:
        main_writer.writeheader()
    count = 0
    IsRun = False
    args.batch_size = batch_size
    file_name = "{}/model_g{}_d{}_lr{}".format(folder_path,g_index,d_index, lr_index)
    print(f"output model:{file_name}")
    args.g_hid = g_hid_value
    args.d_hid = d_hid_value

    args.g_lr = lr_value
    args.d_lr = lr_value
    final_generator, final_discriminator,best_generator, best_discriminator, g_loss, d_loss, stop_epoch, roc_loss = run_model(args,data_loader , file_name)
    #print(f'Epoch {epoch}, D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')
    main_writer.writerow({'id': count, 'g_index': g_index, 'd_index': d_index, 'lr': lr_value, 'g_loss': g_loss, 'd_loss': d_loss, 'stop_epoch': stop_epoch, 'roc': 1-roc_loss})
    print(f'Best model at Epoch {stop_epoch}, best_D_loss: {d_loss:.4f}, best_G_loss: {g_loss:.4f}, roc: {1-roc_loss:.4f}')
    save_model(f'{file_name}_last.pt', 
    {
                    "args": args,
                    "latent_dim": args.latent_dim,
                    "input_dim": args.input_dim,
                    "output_dim": args.output_dim,
                    "init_type": args.init_type,
                    "g_hid": args.g_hid,
                    "d_hid": args.d_hid,
                    "rmse": roc_loss,
                    "g_loss": g_loss,
                    "d_loss": d_loss,
                    "epoch": stop_epoch,
                    "gen_state_dict": final_generator.state_dict(),
                    "dis_state_dict": final_discriminator.state_dict()
    })