import numpy as np
import h5py
from src import tools
import torch
from torch.utils.data import TensorDataset, DataLoader


def save_data(
    RC_mode=True,
):
    """Save data as Tensor from file."""

    with h5py.File("./data/encode_roadmap.h5", "r") as f:
        X_train= f['train_in'][()]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[3])
        X_train = np.array(X_train)
        Y_train = f['train_out'][()]
        Y_train = np.array(Y_train)

        X_val = f['valid_in'][()]
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[3])
        X_val = np.array(X_val)
        Y_val = f['valid_out'][()]
        Y_val = np.array(Y_val)

        X_test = f['test_in'][()]
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[3])
        X_test = np.array(X_test)
        Y_test = f['test_out'][()]
        Y_test = np.array(Y_test)

    if not RC_mode:
        torch.save(torch.from_numpy(X_train).float(), './data/X_train.pt')
        torch.save(torch.from_numpy(Y_train).float(), './data/Y_train.pt')

        torch.save(torch.from_numpy(X_val).float(), './data/X_val.pt')
        torch.save(torch.from_numpy(Y_val).float(), './data/Y_val.pt')

        torch.save(torch.from_numpy(X_test).float(), './data/X_test.pt')
        torch.save(torch.from_numpy(Y_test).float(), './data/Y_test.pt')    
    else:
        X_train_reverse = np.flip(X_train, 2)
        X_train_reverse_complement = np.flip(X_train_reverse, 1)
        X_train = np.concatenate((X_train, X_train_reverse_complement), axis=2)

        X_val_reverse = np.flip(X_val, 2)
        X_val_reverse_complement = np.flip(X_val_reverse, 1)
        X_val = np.concatenate((X_val, X_val_reverse_complement), axis=2)

        X_test_reverse = np.flip(X_test, 2)
        X_test_reverse_complement = np.flip(X_test_reverse, 1)
        X_test = np.concatenate((X_test, X_test_reverse_complement), axis=2)

        torch.save(torch.from_numpy(X_train).float(), './data/X_train_RC.pt')
        torch.save(torch.from_numpy(Y_train).float(), './data/Y_train_RC.pt')

        torch.save(torch.from_numpy(X_val).float(), './data/X_val_RC.pt')
        torch.save(torch.from_numpy(Y_val).float(), './data/Y_val_RC.pt')

        torch.save(torch.from_numpy(X_test).float(), './data/X_test_RC.pt')
        torch.save(torch.from_numpy(Y_test).float(), './data/Y_test_RC.pt')




def load_data2(
    batch_size=256,
    num_workers=12,
    persistent_workers=False,
    RC_mode=True,
):
    """Load data from file."""

    if not RC_mode:
        X_train = torch.load('./data/X_train.pt')
        Y_train = torch.load('./data/Y_train.pt')
        train_dataset = TensorDataset(X_train, Y_train)

        X_val = torch.load('./data/X_val.pt')
        Y_val = torch.load('./data/Y_val.pt')
        val_dataset = TensorDataset(X_val, Y_val)

        X_test = torch.load('./data/X_test.pt')
        Y_test = torch.load('./data/Y_test.pt')
        test_dataset = TensorDataset(X_test, Y_test)
    else:
        X_train = torch.load('./data/X_train_RC.pt')
        Y_train = torch.load('./data/Y_train_RC.pt')
        train_dataset = TensorDataset(X_train, Y_train)

        X_val = torch.load('./data/X_val_RC.pt')
        Y_val = torch.load('./data/Y_val_RC.pt')
        val_dataset = TensorDataset(X_val, Y_val)

        X_test = torch.load('./data/X_test_RC.pt')
        Y_test = torch.load('./data/Y_test_RC.pt')
        test_dataset = TensorDataset(X_test, Y_test)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        persistent_workers=persistent_workers
    )

    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        persistent_workers=persistent_workers
    )

    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        persistent_workers=persistent_workers
    )

    return train_dataloader, val_dataloader, test_dataloader





def load_data(
    file_path,
    batch_size=256,
    num_workers=12,
    persistent_workers=False,
    RC_mode=True,
):
    """Load data from file."""

    with h5py.File(file_path, "r") as f:
        X_train= f['train_in'][()]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[3])
        X_train = np.array(X_train)
        Y_train = f['train_out'][()]
        Y_train = np.array(Y_train)

        X_val = f['valid_in'][()]
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[3])
        X_val = np.array(X_val)
        Y_val = f['valid_out'][()]
        Y_val = np.array(Y_val)

        X_test = f['valid_in'][()]
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[3])
        X_test = np.array(X_test)
        Y_test = f['valid_out'][()]
        Y_test = np.array(Y_test)

    if RC_mode:
        X_train_reverse = np.flip(X_train, 2)
        X_train_reverse_complement = np.flip(X_train_reverse, 1)
        X_train = np.concatenate((X_train, X_train_reverse_complement), axis=2)

        X_val_reverse = np.flip(X_val, 2)
        X_val_reverse_complement = np.flip(X_val_reverse, 1)
        X_val = np.concatenate((X_val, X_val_reverse_complement), axis=2)

        X_test_reverse = np.flip(X_test, 2)
        X_test_reverse_complement = np.flip(X_test_reverse, 1)
        X_test = np.concatenate((X_test, X_test_reverse_complement), axis=2)

    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float())
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        persistent_workers=persistent_workers
    )

    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).float())
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        persistent_workers=persistent_workers
    )

    test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float())
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        persistent_workers=persistent_workers
    ) 

    return train_dataloader, val_dataloader, test_dataloader