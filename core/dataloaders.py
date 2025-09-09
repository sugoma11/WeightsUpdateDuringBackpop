import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_dataloaders(dataset_name, bs):
    """
    Returns train and validation dataloaders for specified dataset

    Args:
        dataset_name: String specifying dataset ('mnist', 'fashion_mnist', or 'cifar10')
        bs: Batch size for the dataloaders

    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    # Define dataset-specific transformations
    if dataset_name.lower() in ['mnist', 'fashion_mnist']:
        # Transformations for MNIST and Fashion MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
    elif dataset_name.lower() == 'cifar10':
        # Transformations for CIFAR10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # CIFAR10 mean and std
        ])
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Use 'mnist', 'fashion_mnist', or 'cifar10'")

    def is_downloaded_mnist(path):
        expected_files = [
            'MNIST/raw/train-images-idx3-ubyte.gz',
            'MNIST/raw/train-labels-idx1-ubyte.gz',
            'MNIST/raw/t10k-images-idx3-ubyte.gz',
            'MNIST/raw/t10k-labels-idx1-ubyte.gz',
        ]
        return all(os.path.exists(os.path.join(path, f)) for f in expected_files)

    def is_downloaded_fashion_mnist(path):
        expected_files = [
            'FashionMNIST/raw/train-images-idx3-ubyte.gz',
            'FashionMNIST/raw/train-labels-idx1-ubyte.gz',
            'FashionMNIST/raw/t10k-images-idx3-ubyte.gz',
            'FashionMNIST/raw/t10k-labels-idx1-ubyte.gz',
        ]
        return all(os.path.exists(os.path.join(path, f)) for f in expected_files)

    def is_downloaded_cifar10(path):
        expected_dir = os.path.join(path, 'cifar-10-batches-py')
        expected_files = [
            'data_batch_1',
            'data_batch_2',
            'data_batch_3',
            'data_batch_4',
            'data_batch_5',
            'test_batch',
            'batches.meta',
        ]
        return os.path.isdir(expected_dir) and all(os.path.exists(os.path.join(expected_dir, f)) for f in expected_files)

    data_dir = '/home/al/projects/experimental/multulayer/data'

    # Dataset loading
    if dataset_name.lower() == 'mnist':
        download = not is_downloaded_mnist(data_dir)
        dataset = datasets.MNIST(data_dir, train=True, download=download, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=download, transform=transform)

    elif dataset_name.lower() == 'fashion_mnist':
        download = not is_downloaded_fashion_mnist(data_dir)
        dataset = datasets.FashionMNIST(data_dir, train=True, download=download, transform=transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=download, transform=transform)

    elif dataset_name.lower() == 'cifar10':
        download = not is_downloaded_cifar10(data_dir)
        dataset = datasets.CIFAR10(data_dir, train=True, download=download, transform=transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=download, transform=transform)

    # Split training data into train and validation sets (80/20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs)
    test_loader = DataLoader(test_dataset, batch_size=bs)

    return train_loader, val_loader, test_loader
