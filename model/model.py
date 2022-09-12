import pandas as pd
import os
import ast
os.environ['AWS_REGION'] = 'us-east-1' # set region, important for sagemaker

# Pytorch and other Image Libraries
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision import models
from torch.utils.data import random_split
from PIL import Image
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from torch.nn import functional as F
from torch import nn

# AWS
from awsio.python.lib.io.s3.s3dataset import S3Dataset
# documentation: https://github.com/aws/amazon-s3-plugin-for-pytorch


class ImageDataset(S3Dataset):
    def __init__(self, s3_bucket_uri: str, transform=None):
        # aws instance and s3 bucket
        s3_session = boto3.Session().resource('s3')
        bucket = s3_session.Bucket(s3_bucket_uri)
        
        # dataframe located in s3 bucket
        bucket.download_file('master.csv', 'master.csv')
        df = pd.read_csv(s3_bucket_uri)
        os.remove('master.csv')
        
        # get filepaths and msrp
        df["Filepaths"] = df["Filepaths"].apply(lambda x: ast.literal_eval(x)).explode()
        filepaths = df["Filepaths"].tolist()
        msrp = [float(m) for m in df["MSRP"].tolist()]
        
        # create dictionary of filepaths and msrp
        self.msrp_dict = {k: v for k, v in zip(filepaths, msrp)}
        self.transform = transform
        super().__init__(filepaths)

    def __getitem__(self, idx):
        _, img = super(ImageDataset, self).__getitem__(idx)
        img = Image.open(self.filepaths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.msrp_dict[self.filepaths[idx]]


class CarDataModule(LightningDataModule):
    def __init__(self, s3_bucket_uri: str, batch_size: int, num_workers: int, train_val_test_split: list, img_shape: tuple):
        super().__init__()
        assert len(train_val_test_split) == 3, "train_val_test_split must be a list of length 3"
        assert sum(train_val_test_split) == 1.0, "train_val_test_split must sum to 1"
        self.s3_bucket_uri = s3_bucket_uri
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_len, self.val_len, self.test_len = train_val_test_split
        self.img_shape = img_shape

    def setup(self, stage=None):
        # transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.img_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        # datasets
        dataset = ImageDataset(self.s3_bucket_uri, self.transform)
        train_size = int(self.train_len * len(dataset))
        val_size = int(self.val_len * len(dataset))
        test_size = len(dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

class CarModule(LightningModule):
    def __init__(self, s3_bucket_uri: str, batch_size: int, num_workers: int, train_val_test_split: list, model):
        super().__init__()
        self.data_module = CarDataModule(s3_bucket_uri, batch_size, num_workers, train_val_test_split, img_shape=model.input_size)
        self.data_module.setup()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()
    

class ResNet18(LightningModule):
    def __init__(self, fine_tune: bool, num_classes: int):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.fc = nn.Linear(512, num_classes)
        self.model.fc.requires_grad = True # Last layer is always trainable
        self.image_size = (224, 224)
    
    def forward(self, x):
        return self.model(x)


class ResNet50(LightningModule):
    def __init__(self, fine_tune: bool, num_classes: int):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.fc = nn.Linear(2048, num_classes)
        self.model.fc.requires_grad = True # Last layer is always trainable
        self.image_size = (224, 224)
    
    def forward(self, x):
        return self.model(x)


class ResNet101(LightningModule):
    def __init__(self, fine_tune: bool, num_classes: int):
        super().__init__()
        self.model = models.resnet101(pretrained=True)
        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.fc = nn.Linear(2048, num_classes)
        self.model.fc.requires_grad = True # Last layer is always trainable
        self.image_size = (224, 224)
    
    def forward(self, x):
        return self.model(x)


class ResNet152(LightningModule):
    def __init__(self, fine_tune: bool, num_classes: int):
        super().__init__()
        self.model = models.resnet152(pretrained=True)
        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.fc = nn.Linear(2048, num_classes)
        self.model.fc.requires_grad = True # Last layer is always trainable
        self.image_size = (224, 224)
    
    def forward(self, x):
        return self.model(x)


class ResNet34(LightningModule):
    def __init__(self, fine_tune: bool, num_classes: int):
        super().__init__()
        self.model = models.resnet34(pretrained=True)
        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.fc = nn.Linear(512, num_classes)
        self.model.fc.requires_grad = True # Last layer is always trainable
        self.image_size = (224, 224)
    
    def forward(self, x):
        return self.model(x)


class VGG11(LightningModule):
    def __init__(self, fine_tune: bool, num_classes: int):
        super().__init__()
        self.model = models.vgg11(pretrained=True)
        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.classifier[6] = nn.Linear(4096, num_classes)
        self.model.classifier[6].requires_grad = True # Last layer is always trainable
        self.image_size = (224, 224)
    
    def forward(self, x):
        return self.model(x)


class VGG13(LightningModule):
    def __init__(self, fine_tune: bool, num_classes: int):
        super().__init__()
        self.model = models.vgg13(pretrained=True)
        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.classifier[6] = nn.Linear(4096, num_classes)
        self.model.classifier[6].requires_grad = True # Last layer is always trainable
        self.image_size = (224, 224)
    
    def forward(self, x):
        return self.model(x)


class VGG16(LightningModule):
    def __init__(self, fine_tune: bool, num_classes: int):
        super().__init__()
        self.model = models.vgg16(pretrained=True)
        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.classifier[6] = nn.Linear(4096, num_classes)
        self.model.classifier[6].requires_grad = True # Last layer is always trainable
        self.image_size = (224, 224)
    
    def forward(self, x):
        return self.model(x)


class VGG19(LightningModule):
    def __init__(self, fine_tune: bool, num_classes: int):
        super().__init__()
        self.model = models.vgg19(pretrained=True)
        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.classifier[6] = nn.Linear(4096, num_classes)
        self.model.classifier[6].requires_grad = True # Last layer is always trainable
        self.image_size = (224, 224)
    
    def forward(self, x):
        return self.model(x)


class AlexNet(LightningModule):
    def __init__(self, fine_tune: bool, num_classes: int):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.classifier[6] = nn.Linear(4096, num_classes)
        self.model.classifier[6].requires_grad = True # Last layer is always trainable
        self.image_size = (256, 256)
    
    def forward(self, x):
        return self.model(x)


class SqueezeNet(LightningModule):
    def __init__(self, fine_tune: bool, num_classes: int):
        super().__init__()
        self.model = models.squeezenet1_0(pretrained=True)
        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.model.classifier[1].requires_grad = True # Last layer is always trainable
        self.image_size = (227, 227)
    
    def forward(self, x):
        return self.model(x)
        

class InceptionV3(LightningModule):
    def __init__(self, fine_tune: bool, num_classes: int):
        super().__init__()
        self.model = models.inception_v3(pretrained=True)
        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.fc = nn.Linear(2048, num_classes)
        self.model.fc.requires_grad = True # Last layer is always trainable
        self.image_size = (299, 299)
    
    def forward(self, x):
        return self.model(x)
    

class DenseNet121(LightningModule):
    def __init__(self, fine_tune: bool, num_classes: int):
        super().__init__()
        self.model = models.densenet121(pretrained=True)
        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.classifier = nn.Linear(1024, num_classes)
        self.model.classifier.requires_grad = True # Last layer is always trainable
        self.image_size = (224, 224)
    
    def forward(self, x):
        return self.model(x)


