'''
Single script that can be run alone and which trains a GPT model on a single household's smart meter data from module 1 SERL outputs
'''

# %%
## imports
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import logging
import sys
from pathlib import Path
import random
import matplotlib.pyplot as plt
# from IPython.display import display, Markdown
import os
import json
import csv
import pickle
import time


# %%
# functions

## experiment set up
def get_next_run_name(base_dir):
    # check existing runs
    existing_runs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('run_')]
    existing_runs.sort() # sort existing run names
    
    if not existing_runs:
        next_run_id = 1
    else:
        # extract run numbers, convert to integer and find the next available
        last_run_id = int(existing_runs[-1].split('_')[1])
        next_run_id = last_run_id + 1
    
    # create new run name, zero padded for sorting consistency
    next_run_name = f'run_{next_run_id:03d}' 
    return next_run_name

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
 
## data pre-processing
### load module 1 data
def load_mod1_data(config):
    logging.info('Selecting data for random PUPRN.')
    pickle_files = [f for f in os.listdir(config['path_to_module_1_data']) if f.endswith('.pkl')]

    continue_to_load_data = True
    while continue_to_load_data:
        random_file = random.choice(pickle_files)
        
        my_puprn = random_file.split('.')[0]
        
        file_path = os.path.join(config['path_to_module_1_data'], random_file)
        
        mod1_data = pd.read_pickle(file_path)
        
        logging.info(f'PUPRN {my_puprn} selected and data loaded.')

        logging.info('Checking for too much missing data...')

        too_much_missing = any((mod1_data[['Clean_elec_imp_Wh','Clean_gas_Wh']].isna().sum() / len(mod1_data)) > config['missing_data_threshold'])
        if too_much_missing:
            logging.info(f'{my_puprn} has too much missing data, going to next choice...')
        else:
            logging.info(f'Missing data check passed for {my_puprn}')
            described = mod1_data[['Clean_elec_imp_Wh','Clean_gas_Wh']].describe()
            count_elec = described.loc['count','Clean_elec_imp_Wh']
            count_gas = described.loc['count','Clean_gas_Wh']
            percent_elec = count_elec / len(mod1_data)
            percent_gas = count_gas / len(mod1_data)
            mean_elec = described.loc['mean','Clean_elec_imp_Wh']
            mean_gas = described.loc['mean','Clean_gas_Wh']
            logging.info(f'Electricity data: count {count_elec}, percent not nan {percent_elec*100}%, mean {mean_elec:.3f}')
            logging.info(f'Gas data: count {count_gas}, percent not nan {percent_gas*100}%, mean {mean_gas:.3f}')
            continue_to_load_data = False
    return mod1_data

### encode / decode data for transformer input / output
def encode_decode_time_series(mod1_data,
                              col:str,
                              config):

    # round energy use values to the nearest 10
    mod1_data[col+'_rounded'] = mod1_data[col].round(config['round_energy_values_to'])
    mod1_data[col+'_rounded_str'] = mod1_data[col+'_rounded'].astype(str).replace('nan','<M>')
    
    # determine the unique values and replace nans with special missing character
    unique_values = np.sort(mod1_data[col+'_rounded'].unique()).astype(str)
    unique_values = np.where(unique_values == 'nan','<M>', unique_values)
    
    if '<M>' not in unique_values:
        unique_values = np.append(unique_values, '<M>')
    
    # determine the vocabulary size
    vocab_size = len(unique_values)
    
    # construct the mappings from character values to indices to index into the Embedding layer
    vtoi = {val: i for i, val in enumerate(unique_values)}
    itov = {i: val for i, val in enumerate(unique_values)}

    encode = lambda v: [vtoi[val] for val in v] # take an interable of values and return a list of indices
    decode = lambda l: [itov[i] for i in l] # take a list of indices and return a list of values

    mod1_data[col+'_rounded_str_encoded'] = encode(mod1_data[col+'_rounded_str'])
    return mod1_data, vocab_size, encode, decode, vtoi

### batchify data
def build_energy_dataset(energy_time_series:np.ndarray,
                         encode,
                         config):
    X, y = [], []
    context = encode(['<M>']) * config['context_length'] # add starting 'missing' tokens to the start
    for i, val in enumerate(energy_time_series[:-1]):
        X.append(context)
        context = context[1:] + [energy_time_series[i]]
        y.append(context)
        
    X = torch.tensor(X)
    y = torch.tensor(y)
    return X, y

def build_calendar_dataset(data,
                           config):
    X_hh, X_dow, X_month = [], [], []
    dows = pd.to_datetime(data['Read_date_effective_local'], format="%Y-%m-%d").dt.dayofweek.values
    months = pd.to_datetime(data['Read_date_effective_local'], format="%Y-%m-%d").dt.month.values
    context_hh = [0] * config['context_length'] # add starting 'missing' tokens to the start
    context_dow = [7] * config['context_length'] # add starting 'missing' tokens to the start, 0 is start day of week so make 7 missing index.
    context_month = [0] * config['context_length'] # add starting 'missing' tokens to the start
    for i, val in enumerate(dows[:-1]):
        X_hh.append(context_hh)
        X_dow.append(context_dow)
        X_month.append(context_month)
        
        context_hh = context_hh[1:] + [data['Readings_from_midnight_local'][i]]
        context_dow = context_dow[1:] + [dows[i]]
        context_month = context_month[1:] + [months[i]]
    X_hh = torch.tensor(X_hh, dtype=torch.float32)
    X_dow = torch.tensor(X_dow, dtype=torch.float32)
    X_month = torch.tensor(X_month, dtype=torch.float32)
    return X_hh, X_dow, X_month

class MultiFeatureDataset(Dataset):
    def __init__(self, input_data, target_data):
        self.input_data = input_data
        self.target_data = target_data
        
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        inputs = self.input_data[idx]
        targets = self.target_data[idx]
        return inputs, targets
    
## training
### GPT model
class GPTSmartMeterModel(nn.Module):
    def __init__(self,
                 vocab_sizes,
                 d_model,
                 nhead,
                 num_layers,
                 dim_feedforward,
                 context_length,
                 dropout=0.1):
        super(GPTSmartMeterModel, self).__init__()
        
        self.num_input_features = len(vocab_sizes)
        self.d_model = d_model
        self.context_length = context_length
        
        # Embedding layers for each input feature (each has its own vocab size)
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size, d_model) for vocab_size in vocab_sizes])
        
        # Learnable positional embedding
        self.pos_embedding = nn.Embedding(context_length, d_model)
        
        # Registering the lower triangular mask as a buffer (for efficiency)
        self.register_buffer('tril_mask', torch.tril(torch.ones(context_length, context_length)))
        
        # Custom Transformer Decoder layers to implement prenorm
        self.transformer_decoder_layers = nn.ModuleList([
            CustomTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)
        ]) 
        
        # output layers for each of the output features (electricity and gas use), assumes elec and gas vocab size are first in vocab_sizes
        self.output_layers = nn.ModuleList([nn.Linear(d_model, vocab_size) for vocab_size in vocab_sizes[0:2]])
        
    def forward(self, x):
        # Shape of x: [num_sequences or batch_size, context_length, num_input_features]
        # print('Input x shape:', x.shape)
        # Embedding each feature separately and summing them up
        embeddings = []
        for i, emb_layer in enumerate(self.embeddings):
            max_index = x[:, :, i].max().item()
            vocab_size = emb_layer.num_embeddings
            if max_index >= vocab_size:
                logging.debug(f'Feature {i} has index out of range. Max index: {max_index}, vocab_size: {vocab_size}')
                raise ValueError(f'Input feature {i} has an index ({max_index}) that exceeds its embedding vocab size ({vocab_size})')
            embeddings.append(emb_layer(x[:, :, i].long())) # [num_sequences, context_length, d_model]
        
        x = sum(embeddings) # [num_sequences, context_length, d_model]
        # print('Embedding x shape:', x.shape)
        
        # Adding learnable positional embedding
        positions = torch.arange(0, self.context_length).unsqueeze(0).expand(x.size(0), -1) # [num_sequences, context_length]
        pos_encoding = self.pos_embedding(positions) # [num_sequences, context_length, d_model]
        x = x + pos_encoding 
        
        # Generate subsequent mask to prevent attending to future values in the sequence
        subsequent_mask = self.tril_mask[:self.context_length, :self.context_length]
        subsequent_mask = subsequent_mask.masked_fill(subsequent_mask == 0, float('-inf'))
        
        # Passing through custom Transformer Decoder layers
        for layer in self.transformer_decoder_layers:
            x = layer(x, tgt_mask = subsequent_mask) # [num_sequences, context_length, d_model]
            
        # Project back to output vocab dimensions (probabilities for elec and gas usage over vocab_size)
        electricity_output = self.output_layers[0](x) # [num_sequences, context_length, vocab_size_elec]
        gas_output = self.output_layers[1](x) # [num_sequences, context_length, vocab_size_gas]
        
        # Return electricity and gas outputs separately
        return electricity_output, gas_output
    
    def generate(self, input_tensor, temperature=1.0):
        """
        Generate new eletricity and gas values.

        Args:
            input_tensor (torch.tensor): Shape [max_length, num_input_features]
            temperature (float): Scaling factor for prediction (higher = more random)
            
        Returns:
        - output_sequence: tensor of shape [max_length, num_input_features]
        """
        self.eval() # set model to evaluation mode
        input_sequence = input_tensor[:self.context_length].unsqueeze(0) # [context_length, num_input_features] -> [1, context_length, num_input_features]
        generated_sequence = input_sequence.clone() # Initialise generated sequence
        output_sequence = input_sequence.clone()
        max_generate_steps = input_tensor.size(0)
        for step in range(self.context_length, max_generate_steps):
            # Forward pass with the current sequence
            electricity_output, gas_output = self.forward(generated_sequence) # [1, context_length, vocab_size_elec], [1, context_length, vocab_size_gas]
            
            # Get the last step's prediction for electricity and gas
            elec_logits = electricity_output[:, -1, :] / temperature
            gas_logits = gas_output[:, -1, :] / temperature
            
            # Apply softmax to get probability distributions
            elec_probs = F.softmax(elec_logits, dim=-1)
            gas_probs = F.softmax(gas_logits, dim=-1)
            
            # Sample from the distributions
            electricity_prediction = torch.multinomial(elec_probs, num_samples=1)
            gas_prediction = torch.multinomial(gas_probs, num_samples=1)
            
            # Create new input feature vector by appending generated predictions
            new_feature_vector = input_tensor[step].clone().unsqueeze(0) # [1, num_input_features]
            new_feature_vector[:, 0] = electricity_prediction # update electricity prediction
            new_feature_vector[:, 1] = gas_prediction # update gas prediction

            # append the new prediction to the generated and output sequences
            generated_sequence = torch.cat([generated_sequence.squeeze(0), new_feature_vector], dim=0) # [context_length + 1, num_input_features]
            output_sequence = torch.cat([output_sequence.squeeze(0), new_feature_vector], dim=0) # [context_length + step + 1, num_input_features]
            
            # prepare generated_sequence for passing through model again
            generated_sequence = generated_sequence.unsqueeze(0) # [1, context_length + 1, num_input_features]
            # Trim generated_sequence to maintain fixed context_length (rolling context window)
            if generated_sequence.size(1) > self.context_length:
                generated_sequence = generated_sequence[:, 1:, :] # [1, context_length, num_input_features]
                
        return output_sequence
    
class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(CustomTransformerDecoderLayer, self).__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, tgt, tgt_mask=None):
        # Prenorm: apply layer normalisation before self-attention
        tgt2 = self.norm1(tgt)
        tgt2, _ = self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        
        # Prenorm: apply layer normalisation before feedforward network
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        
        return tgt
# %%
## main
def main():
    # where will the training run config, log, model and metrics be saved?

    # Find the absolute path of the parent folder, assumed to be the project directory
    project_path = str(Path.cwd().parent)
    
    # add this to the system path for the use of the following processes
    if project_path not in sys.path:
        sys.path.append(project_path)
    
    base_experiment_dir = project_path + '\\experiments\\manual_search\\' # <--- folder where individual training runs should be saved
    
    next_run_name = get_next_run_name(base_experiment_dir)
    
    run_path = os.path.join(base_experiment_dir, next_run_name)
    
    os.makedirs(run_path, exist_ok=True)
    
    # set configuration / hyperparameters
    config = {
        'path_to_module_1_data': 'insert_path_here',
        'input_features': [
            # temperature
            # calendar data (month of year, day of week, hh of day)
        ],
        'target_variable': [
            # 'Clean_gas_kWh_d_mean',
            # 'Clean_elec_imp_kWh_d_mean',
        ],
        'figsize': (15.92/2.54,11.94/2.54),
        'dpi':150,
        'learning_rate': 0.001, 
        'epochs': 50,
        'missing_data_threshold': 0.2,
        'batch_size': 32,
        'context_length': 48*7,
        'eval_iters': 200,
        'n_embd': 32,
        'n_head': 4,
        'n_layer_do': 1,
        'dropout': 0.1,
        'max_demand_threshold': 100000,
        'round_energy_values_to':-1,
        'patience': 20, # Number of epochs to wait before stopping when no val improvement 
        'random_seed': 52,
        'best_model_to_use_for_eval': 'train',
        }

    # save config to run_path
    with open(os.path.join(run_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
        
    # write the initial run details (config only) to the csv file
    summary_file = os.path.join(base_experiment_dir, 'experiment_summary.csv')
    fieldnames = [
        'run_name',
        'puprn',
        'n_embd',
        'learning_rate', 
        'epochs',
        'batch_size',
        'context_length',
        'n_head',
        'n_layer_do',
        'dropout',
        'best_train_loss',
        'best_val_loss', # Placeholder until training commpletes,
        'best_epoch',
        'train_time',
        'patience',
    ]

    # Ensure the summary csv is initialised
    if not os.path.exists(summary_file):
        with open(summary_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    run_details = {key: config[key] for key in fieldnames if key in config}
    run_details.update({
        'run_name': next_run_name,
        'puprn': None,
        'best_train_loss': None,
        'best_val_loss': None, # Placeholder until training commpletes,
        'best_epoch': None,
        'train_time': None,
    })

    # write the initial run details (config only) to the csv file
    with open(summary_file, 'a', newline ='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(run_details)
        
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(),
                                logging.FileHandler(os.path.join(run_path,'logs.txt'), mode='w')
                                ]
                        )
    
    # set the random seed
    set_seed(seed=config['random_seed'])
    
    # load module 1 data for random puprn
    mod1_data = load_mod1_data(config=config)
    
    # set all values less than zero to zero
    if any((mod1_data[['Clean_elec_imp_Wh','Clean_gas_Wh']] < 0).any()):
        logging.info('replace negative energy use values with zero')
        logic = mod1_data['Clean_elec_imp_Wh'] < 0
        mod1_data.loc[logic, 'Clean_elec_imp_Wh'] = 0
        logic = mod1_data['Clean_gas_Wh'] < 0
        mod1_data.loc[logic, 'Clean_gas_Wh'] = 0

    # check for extremely high values
    if any((mod1_data[['Clean_elec_imp_Wh', 'Clean_gas_Wh']] > config['max_demand_threshold']).any()):
        logging.info('set extremely high energy use values to nan')
        logic = mod1_data['Clean_elec_imp_Wh'] > config['max_demand_threshold']
        mod1_data.loc[logic, 'Clean_elec_imp_Wh'] = np.nan
        logic = mod1_data['Clean_gas_Wh'] > config['max_demand_threshold']
        mod1_data.loc[logic, 'Clean_gas_Wh'] = np.nan
        
    # encode electricity data
    mod1_data, vocab_size_elec, encode_elec, decode_elec, vtoi_elec = \
        encode_decode_time_series(mod1_data,
                                'Clean_elec_imp_Wh',
                                config=config)

    # encode gas data
    mod1_data, vocab_size_gas, encode_gas, decode_gas, vtoi_gas = \
        encode_decode_time_series(mod1_data,
                                'Clean_gas_Wh',
                                config=config)
        
    # encode temperature
    # round energy use values to the nearest 10
    mod1_data['temp_C_rounded'] = mod1_data['temp_C'].round(1)
    mod1_data.loc[mod1_data['temp_C_rounded'] == -0.0, 'temp_C_rounded'] = 0.0
    mod1_data['temp_C_rounded_str'] = mod1_data['temp_C_rounded'].astype(str).replace('nan','<M>')


    # determine the unique values and replace nans with special missing character
    unique_values_temp = np.sort(mod1_data['temp_C_rounded'].unique()).astype(str)
    unique_values_temp = np.where(unique_values_temp == 'nan','<M>', unique_values_temp)

    if '<M>' not in unique_values_temp:
        unique_values_temp = np.append(unique_values_temp, '<M>')

    # determine the vocabulary size
    vocab_size_temp = len(unique_values_temp)

    # construct the mappings from character values to indices to index into the Embedding layer
    vtoi_temp = {val: i for i, val in enumerate(unique_values_temp)}
    itov_temp = {i: val for i, val in enumerate(unique_values_temp)}

    encode_temp = lambda v: [vtoi_temp[val] for val in v] # take an interable of values and return a list of indices
    decode_temp = lambda l: [itov_temp[i] for i in l] # take a list of indices and return a list of values

    mod1_data['temp_C_rounded_str_encoded'] = encode_temp(mod1_data['temp_C_rounded_str'])
    
    # split data
    # carve out a random consecutive slice of 20% of the data for val and test splits
    n1 = random.randint(config['context_length'] + 1, len(mod1_data) - int(len(mod1_data) * 0.2))
    n2 = n1 + int(len(mod1_data) * 0.1)
    n3 = n2 + int(len(mod1_data) * 0.1)
    
    # create train / val / test dataframes
    df_train_part1 = mod1_data.iloc[0:n1].copy()
    df_train_part2 = mod1_data.iloc[n3:].copy()
    df_val = mod1_data.iloc[n1:n2].copy()
    df_test = mod1_data.iloc[n2:n3].copy()
    
    # batchify and create tensors
    X_elec_train_part1, y_elec_train_part1 = build_energy_dataset(energy_time_series=df_train_part1['Clean_elec_imp_Wh_rounded_str_encoded'].values, encode=encode_elec, config=config)
    X_elec_train_part2, y_elec_train_part2 = build_energy_dataset(energy_time_series=df_train_part2['Clean_elec_imp_Wh_rounded_str_encoded'].values, encode=encode_elec, config=config)
    X_elec_train = torch.cat([X_elec_train_part1, X_elec_train_part2], dim=0)
    y_elec_train = torch.cat([y_elec_train_part1, y_elec_train_part2], dim=0)
    X_elec_val, y_elec_val = build_energy_dataset(energy_time_series=df_val['Clean_elec_imp_Wh_rounded_str_encoded'].values, encode=encode_elec, config=config)
    X_elec_test, y_elec_test = build_energy_dataset(energy_time_series=df_test['Clean_elec_imp_Wh_rounded_str_encoded'].values, encode=encode_elec, config=config)
    X_elec_gen, y_elec_gen = build_energy_dataset(energy_time_series = mod1_data['Clean_elec_imp_Wh_rounded_str_encoded'].values, encode=encode_elec, config=config)

    X_gas_train_part1, y_gas_train_part1 = build_energy_dataset(energy_time_series=df_train_part1['Clean_gas_Wh_rounded_str_encoded'].values, encode=encode_gas, config=config)
    X_gas_train_part2, y_gas_train_part2 = build_energy_dataset(energy_time_series=df_train_part2['Clean_gas_Wh_rounded_str_encoded'].values, encode=encode_gas, config=config)
    X_gas_train = torch.cat([X_gas_train_part1, X_gas_train_part2], dim=0)
    y_gas_train = torch.cat([y_gas_train_part1, y_gas_train_part2], dim=0)
    X_gas_val, y_gas_val = build_energy_dataset(energy_time_series=df_val['Clean_gas_Wh_rounded_str_encoded'].values, encode=encode_gas, config=config)
    X_gas_test, y_gas_test = build_energy_dataset(energy_time_series=df_test['Clean_gas_Wh_rounded_str_encoded'].values, encode=encode_gas, config=config)
    X_gas_gen, y_gas_gen = build_energy_dataset(energy_time_series = mod1_data['Clean_gas_Wh_rounded_str_encoded'].values, encode=encode_gas, config=config)

    X_temp_train_part1, y_temp_train_part1 = build_energy_dataset(energy_time_series=df_train_part1['temp_C_rounded_str_encoded'].values, encode=encode_temp, config=config)
    X_temp_train_part2, y_temp_train_part2 = build_energy_dataset(energy_time_series=df_train_part2['temp_C_rounded_str_encoded'].values, encode=encode_temp, config=config)
    X_temp_train = torch.cat([X_temp_train_part1, X_temp_train_part2], dim=0)
    y_temp_train = torch.cat([y_temp_train_part1, y_temp_train_part2], dim=0)
    X_temp_val, y_temp_val = build_energy_dataset(energy_time_series=df_val['temp_C_rounded_str_encoded'].values, encode=encode_temp, config=config)
    X_temp_test, y_temp_test = build_energy_dataset(energy_time_series=df_test['temp_C_rounded_str_encoded'].values, encode=encode_temp, config=config)
    X_temp_gen, y_temp_gen = build_energy_dataset(energy_time_series = mod1_data['temp_C_rounded_str_encoded'].values, encode=encode_temp, config=config)
    
    X_hh_train_part1, X_dow_train_part1, X_month_train_part1 = build_calendar_dataset(data=df_train_part1,
                                                                    config=config)
    X_hh_train_part2, X_dow_train_part2, X_month_train_part2 = build_calendar_dataset(data=df_train_part2,
                                                                    config=config)
    X_hh_train = torch.cat([X_hh_train_part1, X_hh_train_part2], dim=0)
    X_dow_train = torch.cat([X_dow_train_part1, X_dow_train_part2], dim=0)
    X_month_train = torch.cat([X_month_train_part1, X_month_train_part2], dim=0)

    X_hh_val, X_dow_val, X_month_val = build_calendar_dataset(data=df_val,
                                                                    config=config)
    X_hh_test, X_dow_test, X_month_test = build_calendar_dataset(data=df_test,
                                                                    config=config)
    X_hh_gen, X_dow_gen, X_month_gen = build_calendar_dataset(data=mod1_data,
                                                                    config=config)
    vocab_size_hh = len(X_hh_train.unique()) + 1 # plus one because we add the 'missing' character
    vocab_size_dow = len(X_dow_train.unique()) + 1
    # vocab_size_month = len(X_month_train.unique()) + 1 
    vocab_size_month = 12 + 1 # need to hard code it, as possible for X_month_train to miss data for a whole month
    
    # we also want to create a DataLoader for the final model with chosen best hyperparameters, which will be trained on the combined train + val data
    X_elec_final_train_and_val = torch.cat([X_elec_train, X_elec_val], dim=0)
    X_gas_final_train_and_val = torch.cat([X_gas_train, X_gas_val], dim=0)
    X_temp_final_train_and_val = torch.cat([X_temp_train, X_temp_val], dim=0)
    X_hh_final_train_and_val = torch.cat([X_hh_train, X_hh_val], dim=0)
    X_dow_final_train_and_val = torch.cat([X_dow_train, X_dow_val], dim=0)
    X_month_final_train_and_val = torch.cat([X_month_train, X_month_val], dim=0)

    y_elec_final_train_and_val = torch.cat([y_elec_train, y_elec_val], dim=0)
    y_gas_final_train_and_val = torch.cat([y_gas_train, y_gas_val], dim=0)
    
    # let's create a concatenated input_tensor which can be used to create a multi feature dataset
    input_tensor_train = torch.stack([X_elec_train, 
                                X_gas_train,
                                X_temp_train,
                                X_hh_train,
                                X_dow_train,
                                X_month_train],
                                    dim=2)
    input_tensor_val = torch.stack([X_elec_val, 
                                X_gas_val,
                                X_temp_val,
                                X_hh_val,
                                X_dow_val,
                                X_month_val],
                                dim=2)
    input_tensor_test = torch.stack([X_elec_test, 
                                X_gas_test,
                                X_temp_test,
                                X_hh_test,
                                X_dow_test,
                                X_month_test]
                                    ,dim=2)
    input_tensor_final_train_and_val = torch.stack([X_elec_final_train_and_val, 
                                X_gas_final_train_and_val,
                                X_temp_final_train_and_val,
                                X_hh_final_train_and_val,
                                X_dow_final_train_and_val,
                                X_month_final_train_and_val]
                                    ,dim=2)
    input_tensor_gen = torch.stack([X_elec_gen, 
                                X_gas_gen,
                                X_temp_gen,
                                X_hh_gen,
                                X_dow_gen,
                                X_month_gen]
                                    ,dim=2)

    target_tensor_train = torch.stack([y_elec_train,
                                    y_gas_train],
                                    dim=2)
    target_tensor_val = torch.stack([y_elec_val,
                                    y_gas_val],
                                    dim=2)
    target_tensor_test = torch.stack([y_elec_test,
                                    y_gas_test],
                                    dim=2)
    target_tensor_final_train_and_val = torch.stack([y_elec_final_train_and_val,
                                    y_gas_final_train_and_val],
                                    dim=2)
    
    # create multifeature datasets (for data loaders)
    dataset_train = MultiFeatureDataset(input_data = input_tensor_train,
                                        target_data = target_tensor_train)
    dataset_val = MultiFeatureDataset(input_data = input_tensor_val,
                                    target_data = target_tensor_val)
    dataset_test = MultiFeatureDataset(input_data = input_tensor_test,
                                    target_data = target_tensor_test)
    dataset_final_train_and_val = MultiFeatureDataset(input_data = input_tensor_final_train_and_val,
                                    target_data = target_tensor_final_train_and_val)
    
    # create DataLoaders
    dataloader_train = DataLoader(dataset=dataset_train,
                                batch_size=config['batch_size'],
                                shuffle=True)
    dataloader_val = DataLoader(dataset = dataset_val,
                                batch_size= config['batch_size'],
                                shuffle = True)
    dataloader_test = DataLoader(dataset = dataset_test,
                                batch_size = config['batch_size'],
                                shuffle = True)
    dataloader_final_train_and_val = DataLoader(dataset = dataset_final_train_and_val,
                                batch_size = config['batch_size'],
                                shuffle = True)
    
    # save to disk
    # input and target tensors (can be used to create DataLoaders again)
    torch.save(input_tensor_train, os.path.join(run_path, 'input_tensor_train.pt'))
    torch.save(input_tensor_val, os.path.join(run_path, 'input_tensor_val.pt'))
    torch.save(input_tensor_test, os.path.join(run_path, 'input_tensor_test.pt'))
    torch.save(input_tensor_final_train_and_val, os.path.join(run_path, 'input_tensor_final_train_and_val.pt'))
    torch.save(input_tensor_gen, os.path.join(run_path, 'input_tensor_gen.pt'))

    torch.save(target_tensor_train, os.path.join(run_path, 'target_tensor_train.pt'))
    torch.save(target_tensor_val, os.path.join(run_path, 'target_tensor_val.pt'))
    torch.save(target_tensor_test, os.path.join(run_path, 'target_tensor_test.pt'))
    torch.save(target_tensor_final_train_and_val, os.path.join(run_path, 'target_tensor_final_train_and_val.pt'))
    
    # record puprn that has been selected
    df = pd.read_csv(summary_file)
    my_puprn = mod1_data['PUPRN'].unique()[0]
    df.loc[df['run_name'] == next_run_name, 'puprn'] = my_puprn
    df.to_csv(summary_file, index=False)
    
    # save dataframes (train, val, test)
    with open(os.path.join(run_path, 'df_train_part1.pkl'), 'wb') as f:
        pickle.dump(df_train_part1, f)

    with open(os.path.join(run_path, 'df_train_part2.pkl'), 'wb') as f:
        pickle.dump(df_train_part2, f)
        
    with open(os.path.join(run_path, 'df_train_val.pkl'), 'wb') as f:
        pickle.dump(df_val, f)
        
    with open(os.path.join(run_path, 'df_test.pkl'), 'wb') as f:
        pickle.dump(df_test, f)
        
    # save tokenisers
    vocab_elec = vtoi_elec
    with open(os.path.join(run_path, 'vocab_elec.pkl'), 'wb') as f:
        pickle.dump(vocab_elec, f)

    vocab_gas = vtoi_gas
    with open(os.path.join(run_path, 'vocab_gas.pkl'), 'wb') as f:
        pickle.dump(vocab_gas, f)
        
    vocab_temp = vtoi_temp
    with open(os.path.join(run_path, 'vocab_temp.pkl'), 'wb') as f:
        pickle.dump(vocab_temp, f)
        
    # training
    vocab_sizes = [vocab_size_elec,
                vocab_size_gas,
                vocab_size_temp,
                vocab_size_hh,
                vocab_size_dow,
                vocab_size_month]
    # instantiate the model
    model = GPTSmartMeterModel(vocab_sizes=vocab_sizes,
                            d_model=config['n_embd'],
                            nhead=config['n_head'],
                            num_layers=config['n_layer_do'],
                            dim_feedforward=config['n_embd'] * 2,
                            context_length=config['context_length'],
                            dropout=config['dropout'])

    # logging.info the number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters())/1e6
    logging.info(f'{num_params} M parameters')

    # create a PyTorch optimiser
    optim = torch.optim.AdamW(model.parameters(), lr = config['learning_rate'])

    # create a loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # training loop
    epochi = []
    stepi = []
    step_train_lossi = []
    epoch_train_lossi = []
    epoch_val_lossi = []
    counter = 0
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_epoch = np.nan
    epochs_without_improvement = 0
    best_val_model_path = os.path.join(run_path,'best_val_model.pth')
    best_train_model_path = os.path.join(run_path,'best_train_model.pth')
    start_time = time.time()

    for epoch in range(0,config['epochs']):
        logging.info(f'starting training epoch {epoch}')
        model.train()
        epochi.append(epoch)
        total_loss = 0
        
        for i, (X, y) in enumerate(dataloader_train):
            counter += 1
            stepi.append(counter)
            # forward pass
            
            electricity_output, gas_output = model(X) # [batch_size, context_length, vocab_size_elec], [batch_size, context_length, vocab_size_gas]
            
            # calculate the loss (flatten output and target for compatibility with CrossEntropyLoss)
            electricity_output = electricity_output.view(-1, electricity_output.size(-1)) # [batch_size * context_length, vocab_size_elec]
            gas_output = gas_output.view(-1, gas_output.size(-1))
            
            electricity_target = y[:, :, 0].view(-1) # [batch_size * context_length]
            gas_target = y[:, :, 1].view(-1)
            
            electricity_loss = loss_fn(electricity_output, electricity_target)
            gas_loss = loss_fn(gas_output, gas_target)
            
            loss = (electricity_loss + gas_loss) / 2
            
            step_train_lossi.append(loss.item())
            total_loss += loss.item()
            
            # backward 
            optim.zero_grad()
            loss.backward()
            optim.step()
        avg_loss = total_loss / len(dataloader_train)
        epoch_train_lossi.append(avg_loss)
        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            # save the best performing model on train loss
            torch.save(model.state_dict(), best_train_model_path) # save best val model
            
        logging.info(f'Training loss: {avg_loss}')
        # after each epoch, evaluate the val loss
        model.eval()
        with torch.no_grad():
            val_lossi = []
            for vali in range(0,config['eval_iters']):
                # get a batch
                X_val, y_val = next(iter(dataloader_val))
                
                # forward pass
                electricity_output, gas_output = model(X_val)
                
                # calculate val loss
                electricity_output = electricity_output.view(-1, electricity_output.size(-1)) # [batch_size * context_length, vocab_size_elec]
                gas_output = gas_output.view(-1, gas_output.size(-1))
                
                electricity_target = y_val[:, :, 0].view(-1) # [batch_size * context_length]
                gas_target = y_val[:, :, 1].view(-1)
                
                electricity_loss = loss_fn(electricity_output, electricity_target)
                gas_loss = loss_fn(gas_output, gas_target)
                
                val_loss = (electricity_loss + gas_loss) / 2
                val_lossi.append(val_loss.item())
        epoch_val_lossi.append(np.mean(val_lossi))
        
        logging.info(f'Val loss: {np.mean(val_lossi)}')
        
        # early stopping logic
        if np.mean(val_lossi) < best_val_loss:
            best_val_loss = np.mean(val_lossi)
            epochs_without_improvement = 0
            best_epoch = epoch
            torch.save(model.state_dict(), best_val_model_path) # save best val model
            logging.info('Validation loss improved, saving best model weights.')
        else: 
            epochs_without_improvement += 1
            logging.info(f'No improvement in validation for {epochs_without_improvement} epochs.')    
        
        if epochs_without_improvement >= config['patience']:
            patience = config['patience']
            logging.info(f'Stopping early after {patience} epochs without improvement.')
            break    
        
        model.train()
        
    train_time = time.time() - start_time
    
    # save training metrics
    # save run metrics
    metrics = {
        'best_train_loss': best_train_loss,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'train': train_time
    }
    # record metrics
    df = pd.read_csv(summary_file)
    df.loc[df['run_name'] == next_run_name, \
        ['best_train_loss', 'best_val_loss', 'best_epoch', 'train_time']] = \
            [best_train_loss, best_val_loss, best_epoch, train_time]
    df.to_csv(summary_file, index=False)
    
    # plot and save losses
    fig, ax = plt.subplots(1,1,figsize=config['figsize'])
    colors = ['b','g','r','c','k','y','m']
    # for i in range(0,1):
    i = 1
    color = colors[i % len(colors)]

    ax.plot(epochi, 
                epoch_train_lossi,
                label='Training loss',
                color = color)
    ax.plot(epochi, 
                epoch_val_lossi,
                linestyle='--',
                label='Val loss',
                color=color)
    ax.set_xlabel('epochs')
    ax.set_ylabel('Loss')
    ax.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    # plt.show()
    plot_data = {
        'epochi': epochi,
        'epoch_train_lossi': epoch_train_lossi,
        'epoch_val_lossi': epoch_val_lossi, 
    }
    plot_data = pd.DataFrame(data=plot_data)
    plots_folder = os.path.join(run_path, 'plots')
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder, exist_ok=True)

    path_to_fig = os.path.join(plots_folder, 'train_val_loss.png')
    fig.savefig(path_to_fig,dpi=config['dpi'],bbox_inches='tight')
    plot_data.to_csv(os.path.join(run_path, 'train_val_loss.csv'),index=False)

    # inference
    # load best model
    if config['best_model_to_use_for_eval'] == 'best_train_model':
        logging.info('Evaluating on best train model')
        if os.path.exists(best_train_model_path):
            model.load_state_dict(torch.load(best_train_model_path))
            logging.info('Loaded best train model weights from training run')
    elif config['best_model_to_use_for_eval'] == 'best_val_model':
        logging.info('Evaluating on best val model')
        if os.path.exists(best_val_model_path):
            model.load_state_dict(torch.load(best_val_model_path))
            logging.info('Loaded best val model weights from training run')
        
    # generate
    # [:, 0, :] -> [len_tensor, num_input_features]
    generated_sequence = model.generate(input_tensor_gen[:,0,:])
    
    # create a dataframe of generated data
    # we want to skip the first context_length values, as they are nan
    starting_index = config['context_length']
    # electricity
    decoded = np.array(decode_elec(generated_sequence[starting_index:,0].tolist()))
    generated_elec = pd.Series((np.where(decoded == '<M>', 'nan', decoded)).astype(float), name='Clean_elec_imp_Wh')

    # gas 
    decoded = np.array(decode_gas(generated_sequence[starting_index:,1].tolist()))
    generated_gas = pd.Series((np.where(decoded == '<M>', 'nan', decoded)).astype(float), name='Clean_gas_Wh')

    # temperature
    decoded = np.array(decode_temp(generated_sequence[starting_index:,2].tolist()))
    generated_temp = pd.Series((np.where(decoded == '<M>', 'nan', decoded)).astype(float), name='temp_C')

    # half-hour of day
    decoded = np.array(generated_sequence[starting_index:,3].tolist())
    generated_hh = pd.Series((np.where(decoded == 0, 'nan', decoded)).astype(float), name='Readings_from_midnight_local')

    # day of the week
    decoded = np.array(generated_sequence[starting_index:,4].tolist())
    generated_dow = pd.Series((np.where(decoded == 7, 'nan', decoded)).astype(float), name='dayofweek')

    # month of year
    decoded = np.array(generated_sequence[starting_index:,5].tolist())
    generated_month = pd.Series((np.where(decoded == 0, 'nan', decoded)).astype(float), name='month')

    df_generated = pd.DataFrame({generated_elec.name: generated_elec,
                                generated_gas.name: generated_gas,
                                generated_temp.name: generated_temp,
                                generated_hh.name: generated_hh,
                                generated_dow.name: generated_dow,
                                generated_month.name: generated_month})
    df_generated = df_generated[[
        'month',
        'dayofweek',
        'Readings_from_midnight_local',
        'Clean_elec_imp_Wh',
        'Clean_gas_Wh',
        'temp_C'
    ]]
    
    # save generated df to disk
    generated_data_path = os.path.join(run_path,f'generated_data_{my_puprn}.csv')
    df_generated.to_csv(generated_data_path, index=False)
    
    # evaluate
    # here we will use the tensor used to generate the synthetic data - we'll call this the prompt
    tensor_prompt = input_tensor_gen[:,0,:]
    # we want to skip the first context_length values, as they are nan
    starting_index = config['context_length']
    # electricity
    decoded = np.array(decode_elec(tensor_prompt[starting_index:,0].tolist()))
    prompt_elec = pd.Series((np.where(decoded == '<M>', 'nan', decoded)).astype(float), name='Clean_elec_imp_Wh')

    # gas 
    decoded = np.array(decode_gas(tensor_prompt[starting_index:,1].tolist()))
    prompt_gas = pd.Series((np.where(decoded == '<M>', 'nan', decoded)).astype(float), name='Clean_gas_Wh')

    # temperature
    decoded = np.array(decode_temp(tensor_prompt[starting_index:,2].tolist()))
    prompt_temp = pd.Series((np.where(decoded == '<M>', 'nan', decoded)).astype(float), name='temp_C')

    # half-hour of day
    decoded = np.array(tensor_prompt[starting_index:,3].tolist())
    prompt_hh = pd.Series((np.where(decoded == 0, 'nan', decoded)).astype(float), name='Readings_from_midnight_local')

    # day of the week
    decoded = np.array(tensor_prompt[starting_index:,4].tolist())
    prompt_dow = pd.Series((np.where(decoded == 7, 'nan', decoded)).astype(float), name='dayofweek')

    # month of year
    decoded = np.array(tensor_prompt[starting_index:,5].tolist())
    prompt_month = pd.Series((np.where(decoded == 0, 'nan', decoded)).astype(float), name='month')

    df_prompt = pd.DataFrame({prompt_elec.name: prompt_elec,
                                prompt_gas.name: prompt_gas,
                                prompt_temp.name: prompt_temp,
                                prompt_hh.name: prompt_hh,
                                prompt_dow.name: prompt_dow,
                                prompt_month.name: prompt_month})
    df_prompt = df_prompt[[
        'month',
        'dayofweek',
        'Readings_from_midnight_local',
        'Clean_elec_imp_Wh',
        'Clean_gas_Wh',
        'temp_C'
    ]]
    
    # save prompt df to disk
    prompt_data_path = os.path.join(run_path,f'prompt_data_{my_puprn}.csv')
    df_prompt.to_csv(prompt_data_path, index=False)
    
    # eval plots
    # distributions
    fig, ax = plt.subplots(2,2, figsize=config['figsize'], dpi=config['dpi'])

    # original elec
    data = df_prompt['Clean_elec_imp_Wh']
    ax[0,0].hist(data, color='black', alpha=0.5, bins=30, label='Original')

    # original gas
    data = df_prompt['Clean_gas_Wh']
    ax[0,1].hist(data, color='black', alpha=0.5, bins= 30, label='Original')

    # generated elec
    data = df_generated['Clean_elec_imp_Wh']
    ax[1,0].hist(data, color='gray', alpha=0.5, bins=30, label='Generated')

    # generated gas
    data = df_generated['Clean_gas_Wh']
    ax[1,1].hist(data, color='gray', alpha=0.5, bins=30, label='Generated')

    ax[0,0].set_title('Electricity')
    ax[0,1].set_title('Gas')
    ax[0,0].set_ylabel('Frequency')
    ax[0,1].set_ylabel('Frequency')
    ax[1,0].set_ylabel('Frequency')
    ax[1,1].set_ylabel('Frequency')
    ax[1,0].set_xlabel('Wh / half-hour')
    ax[1,1].set_xlabel('Wh / half-hour')

    ax[0,0].legend()
    ax[0,1].legend()
    ax[1,0].legend()
    ax[1,1].legend()
    ax[0,0].grid()
    ax[0,1].grid()
    ax[1,0].grid()
    ax[1,1].grid()

    plt.tight_layout()
    plots_folder = os.path.join(run_path, 'plots')
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder, exist_ok=True)

    path_to_fig = os.path.join(plots_folder, 'distributions.png')
    fig.savefig(path_to_fig,dpi=config['dpi'],bbox_inches='tight')

    # time series visualisation
    fig, ax = plt.subplots(2,2, figsize=config['figsize'], dpi=config['dpi'])
    start = 0
    num_days = 7
    end = start + 48 * num_days
    ax[0,0].plot(df_prompt['Clean_elec_imp_Wh'][start:end], color='b', label='Original')

    ax[1,0].plot(df_generated['Clean_elec_imp_Wh'][start:end], color='b', label='Generated')

    ax[0,1].plot(df_prompt['Clean_gas_Wh'][start:end], color='g', label='Original')

    ax[1,1].plot(df_generated['Clean_gas_Wh'][start:end], color='g', label='Generated')

    ax[0,0].set_title('Electricity')
    ax[0,1].set_title('Gas')
    ax[0,0].set_ylabel('Wh')
    ax[0,1].set_ylabel('Wh')
    ax[1,0].set_ylabel('Wh')
    ax[1,1].set_ylabel('Wh')

    ax[0,0].set_xticklabels([])
    ax[0,1].set_xticklabels([])

    ax[1,0].set_xlabel('Half-hour')
    ax[1,1].set_xlabel('Half-hour')

    ax[0,0].legend()
    ax[1,0].legend()
    ax[0,1].legend()
    ax[1,1].legend()
    plt.tight_layout()

    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder, exist_ok=True)

    path_to_fig = os.path.join(plots_folder, 'time_series.png')
    fig.savefig(path_to_fig,dpi=config['dpi'],bbox_inches='tight')

    # demand profile (avergae)
    fig, ax = plt.subplots(1,2, figsize=config['figsize'], dpi = config['dpi'])

    # electricity
    original_demand_profile = df_prompt[['Readings_from_midnight_local','Clean_elec_imp_Wh']].groupby(by='Readings_from_midnight_local').mean()
    generated_demand_profile = df_generated[['Readings_from_midnight_local','Clean_elec_imp_Wh']].groupby(by='Readings_from_midnight_local').mean()

    ax[0].plot(original_demand_profile, color='b', label='Original')
    ax[0].plot(generated_demand_profile.loc[1:], color='b', linestyle='--', label='Generated')
    ax[0].set_title('Electricity')
    ax[0].set_ylabel('Wh')
    ax[0].set_xlabel('Half-hour of day')
    ax[0].legend()

    # gas
    original_demand_profile = df_prompt[['Readings_from_midnight_local','Clean_gas_Wh']].groupby(by='Readings_from_midnight_local').mean()
    generated_demand_profile = df_generated[['Readings_from_midnight_local','Clean_gas_Wh']].groupby(by='Readings_from_midnight_local').mean()

    ax[1].plot(original_demand_profile, color='g', label='Original')
    ax[1].plot(generated_demand_profile.loc[1:], color='g', linestyle='--', label='Generated')
    ax[1].set_title('Gas')
    ax[1].set_ylabel('Wh')
    ax[1].set_xlabel('Half-hour of day')
    ax[1].legend()

    plt.tight_layout()

    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder, exist_ok=True)

    path_to_fig = os.path.join(plots_folder, 'demand_profile.png')
    fig.savefig(path_to_fig,dpi=config['dpi'],bbox_inches='tight')

    # energy vs ext temperature (PTG)
    fig, ax = plt.subplots(1,1, figsize=config['figsize'], dpi = config['dpi'])

    grouped = df_generated[['month','temp_C','Clean_gas_Wh']].groupby(by='month').mean()
    grouped = grouped.dropna()
    ax.scatter(grouped['temp_C'], grouped['Clean_gas_Wh'], color='g', label='Generated')

    temp = df_prompt.copy()
    grouped = temp[['month','temp_C','Clean_gas_Wh']].groupby(by='month').mean()
    grouped = grouped.dropna()
    ax.scatter(grouped['temp_C'], grouped['Clean_gas_Wh'], color='b', label='Original')

    ax.legend()
    ax.set_ylabel('Wh average per month')
    ax.set_xlabel('Temperature average per month')
    ax.set_title('Gas use vs external temperature')
    plt.tight_layout()

    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder, exist_ok=True)

    path_to_fig = os.path.join(plots_folder, 'ptg.png')
    fig.savefig(path_to_fig,dpi=config['dpi'],bbox_inches='tight')

if __name__ == "__main__":
    main()