import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset 

def make_dataset(L, H, Y, U, KFpredX, batch_size=64):
    ar_data_inputs = []
    ar_data_outputs = []
    ar_data_kf_states = []

    T = Y.shape[0]-1

    for t in range(L, T-max(L,H)):
        input_features = np.concatenate([
            Y[t-L:t][::-1].flatten(order='C'),
            U[t-L:t][::-1].flatten(order='C')
        ])
        ar_data_inputs.append(input_features)

        ar_data_outputs.append(Y[t:t+H, :].flatten(order='C'))

        ar_data_kf_states.append(KFpredX[t, :].flatten(order='C'))

    ar_data_inputs_np = np.array(ar_data_inputs)
    ar_data_outputs_np = np.array(ar_data_outputs)
    ar_data_kf_states_np = np.array(ar_data_kf_states)

    ar_inputs_tensor = torch.tensor(ar_data_inputs_np, dtype=torch.float32)
    ar_outputs_tensor = torch.tensor(ar_data_outputs_np, dtype=torch.float32)

    dataset = TensorDataset(ar_inputs_tensor, ar_outputs_tensor)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, ar_data_inputs_np, ar_data_outputs_np, ar_data_kf_states_np