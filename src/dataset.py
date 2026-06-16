import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
NUM_CLASSES = 7


class FERDataset(Dataset):
    def __init__(self, df, transform=None):
        self.pixels = df['pixels'].tolist()
        self.labels = df['emotion'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pixels = np.array(self.pixels[idx].split(), dtype=np.uint8)
        image = pixels.reshape(48, 48)
        image = Image.fromarray(image, mode='L')

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


def get_transforms(augment=True, for_transfer=False):
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    if for_transfer:
        train_t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return train_t, val_t

    if augment:
        train_t = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomCrop(48, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_t = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    val_t = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    return train_t, val_t


def get_dataloaders(csv_path, batch_size=64, augment=True, for_transfer=False, num_workers=2):
    import numpy as np
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path)

    # emotion -> int; drop stray non-numeric rows (e.g. an 'ing' fragment)
    df['emotion'] = pd.to_numeric(df['emotion'], errors='coerce')
    df = df.dropna(subset=['emotion', 'pixels'])
    df['emotion'] = df['emotion'].astype(int)

    # drop malformed pixel rows so reshape(48,48) can never crash mid-training
    df = df[df['pixels'].str.split().str.len() == 2304].reset_index(drop=True)

    # normalize Usage; if it's missing or collapsed to one split, build our own
    valid = ['Training', 'PublicTest', 'PrivateTest']
    has_usage = ('Usage' in df.columns and
                 df['Usage'].astype(str).str.strip().isin(valid).any())
    if has_usage:
        df['Usage'] = df['Usage'].astype(str).str.strip()
        train_df = df[df['Usage'] == 'Training'].reset_index(drop=True)
        val_df   = df[df['Usage'] == 'PublicTest'].reset_index(drop=True)
        test_df  = df[df['Usage'] == 'PrivateTest'].reset_index(drop=True)
        if len(val_df) == 0 or len(test_df) == 0:
            has_usage = False
    if not has_usage:
        train_df, tmp = train_test_split(df, test_size=0.2, random_state=42,
                                         stratify=df['emotion'])
        val_df, test_df = train_test_split(tmp, test_size=0.5, random_state=42,
                                           stratify=tmp['emotion'])
        train_df, val_df, test_df = (x.reset_index(drop=True)
                                     for x in (train_df, val_df, test_df))

    assert len(val_df) > 0, "Validation set is empty — check the CSV's Usage column / row count"

    train_t, val_t = get_transforms(augment=augment, for_transfer=for_transfer)

    train_ds = FERDataset(train_df, transform=train_t)
    val_ds = FERDataset(val_df, transform=val_t)
    test_ds = FERDataset(test_df, transform=val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


def get_class_weights(csv_path):
    df = pd.read_csv(csv_path)
    train_df = df[df['Usage'] == 'Training']
    counts = train_df['emotion'].value_counts().sort_index().values
    weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float)
