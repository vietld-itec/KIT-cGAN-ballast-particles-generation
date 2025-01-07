import pandas as pd
import torch
import numpy as np
import csv

# Define a function to generate real polygon profiles
def generate_real_data(args):
  # Assume that the polygon profiles are normalized to the range [-1, 1]
  #data = np.random.uniform(-1, 1, (batch_size, output_dim))
  #data = torch.from_numpy(data).float().to(device)
    with open(args.path_file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    data = np.array(data, dtype=float)
    SAMPLE_LEN = args.sample_len       # number N of points where a curve is sampled
    SAMPLE_SIZE = args.sample_size   # number of curves in the training set

    STARTID = args.sample_startid

    SAMPLE = np.zeros((SAMPLE_SIZE, SAMPLE_LEN))
    args.batch_size = SAMPLE_SIZE
    for i in range(STARTID, STARTID+SAMPLE_SIZE):
        #rint(i)
        SAMPLE[i-STARTID] = data[i,1:]
    return torch.from_numpy(SAMPLE).float().to(args.device)

# Define a function to generate real polygon profiles
def generate_random_data(args):
  # Assume that the polygon profiles are normalized to the range [-1, 1]
  #data = np.random.uniform(-1, 1, (batch_size, output_dim))
  #data = torch.from_numpy(data).float().to(device)
    with open(args.path_file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    data = np.array(data, dtype=float)
    SAMPLE_LEN = args.sample_len       # number N of points where a curve is sampled
    SAMPLE_SIZE = args.sample_size   # number of curves in the training set

    STARTID = args.sample_startid

    SAMPLE = np.zeros((SAMPLE_SIZE, SAMPLE_LEN))
    args.batch_size = SAMPLE_SIZE
    for i in range(STARTID, STARTID+SAMPLE_SIZE):
        #rint(i)
        SAMPLE[i-STARTID] = data[i,1:]
    # Create a vector representing radial distances
    num_values = args.sample_len
    
    # Choose a random starting point
    #random_start_index = np.random.randint(num_values)
    #print(f"random_start:{random_start_index}")
    # Choose a random starting point for each row
    random_start_indices = np.random.randint(SAMPLE_LEN, size=SAMPLE_SIZE)
    #print(f"random_start_indices: {random_start_indices}")

    # Rotate each row independently
    rotated_values = np.array([np.roll(row, start_index) for row, start_index in zip(SAMPLE, random_start_indices)])

    # Rotate the vector by circularly shifting the values
    #rotated_values = np.concatenate([SAMPLE[random_start_index:-1],  SAMPLE[0:random_start_index]]) #np.roll(radial_values, random_start_index)
    #rotated_values = np.concatenate([SAMPLE[random_start_index:], SAMPLE[:random_start_index+1]])
    return torch.from_numpy(rotated_values).float().to(args.device)


def random_batch_data(data, batchsize):
    data = data.cpu().numpy()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # The device to use
    # Generate random start indices for rotation
    random_start_indices = np.random.randint(len(data), size=batchsize)
    #print("Random start indices:", random_start_indices)

    # Rotate each sample independently
    rotated_values = np.array([np.roll(row, start_index) for row, start_index in zip(data, random_start_indices)])
    #print("Rotated values:")
    #print(rotated_values)
    out_values = torch.from_numpy(rotated_values).float().to(device)
    return out_values


def random_data(data):
    data = data.cpu().numpy()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # The device to use
    # Generate random start indices for rotation
    random_start_indices = np.random.randint(len(data), size=1)
    #print("Random start indices:", random_start_indices)

    # Rotate each sample independently
    rotated_values = np.array([np.roll(data, start_index) for start_index in random_start_indices])
    #print("Rotated values:")
    #print(rotated_values)
    out_values = torch.from_numpy(rotated_values).float().to(device)
    return out_values[0]

# Define a function to generate random noise vectors
def generate_noise(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # The device to use
    noise = torch.randn(args.batch_size, args.latent_dim, device=device)
    return noise

# Define a function to generate random noise vectors
def generate_noise_vector(args):
    noise = torch.randn(1, args.latent_dim, device=args.device)
    return noise
