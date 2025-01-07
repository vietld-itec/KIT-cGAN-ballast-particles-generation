import numpy as np
import os
import csv
import torch
import yaml
from torch.utils.data import DataLoader

# Placeholder imports
# Replace these with actual class or function implementations
from your_library import CustomArgs, CustomDataset, run_model, save_model

def load_config(yaml_path):
    """Load configuration from a YAML file."""
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(config_path, g_index, d_index, lr_index, nsamples=360):
    # Load configuration from YAML
    config = load_config(config_path)

    g_hid = np.array(config['g_hid'])
    d_hid = np.array(config['d_hid'])
    lr_range = np.array(config['lr_range'])

    # Select model parameters
    g_hid_value = g_hid[g_index]
    d_hid_value = d_hid[d_index]
    lr_value = lr_range[lr_index]

    # Initialize arguments
    args = CustomArgs()
    folder_path = f'selected-vcgan-{nsamples}'

    args.path_file = f'csv/csvdat{nsamples}plus.csv'
    args.sample_len = nsamples
    args.sample_size = 40
    args.sample_startid = 0
    args.classes = ['Angular', 'Rounded', 'Subangular', 'Subrounded']
    args.record_file = 'loss.csv'
    args.g_hid = g_hid_value
    args.d_hid = d_hid_value
    args.num_classes = 4

    args.init_type = "xavier_uniform"
    args.kernel = "rbf"
    args.g_lr = lr_value
    args.d_lr = lr_value
    args.beta1 = 0.5
    args.beta2 = 0.999
    args.input_dim = nsamples
    args.output_dim = nsamples
    args.latent_dim = 100
    args.n_critic = 1

    args.num_epochs = 500000
    args.patience = 50000
    args.min_epochs = 50000
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args.row = 5
    args.col = 8
    args.batch_size = 40

    # Create output folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

    main_csv_file = f"{folder_path}/combination.csv"
    dataset = CustomDataset(args.path_file)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Open the CSV file for writing or appending
    with open(main_csv_file, 'a', newline='') as csvfile:
        fieldnames = ['id', 'g_index', 'd_index', 'lr', 'g_loss', 'd_loss', 'stop_epoch', 'roc']
        main_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if file is empty
        if csvfile.tell() == 0:
            main_writer.writeheader()

        count = 0
        file_name = f"{folder_path}/model_g{g_index}_d{d_index}_lr{lr_index}"
        print(f"Output model: {file_name}")

        # Run the model
        final_generator, final_discriminator, best_generator, best_discriminator, g_loss, d_loss, stop_epoch, roc_loss = run_model(args, data_loader, file_name)

        main_writer.writerow({
            'id': count,
            'g_index': g_index,
            'd_index': d_index,
            'lr': lr_value,
            'g_loss': g_loss,
            'd_loss': d_loss,
            'stop_epoch': stop_epoch,
            'roc': 1 - roc_loss
        })

        print(f"Best model at Epoch {stop_epoch}, best_D_loss: {d_loss:.4f}, best_G_loss: {g_loss:.4f}, roc: {1 - roc_loss:.4f}")

        # Save the model
        save_model(f'{file_name}_last.pt', {
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

# Example usage
# Assuming a config YAML file exists with g_hid, d_hid, and lr_range defined
# Example config_path: 'config.yaml'
# main('config.yaml', g_index=3, d_index=6, lr_index=0)
