import argparse
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from model.model import CarModule, ResNet101, ResNet152, ResNet18, ResNet34, ResNet50, VGG11, VGG13, VGG16, VGG19, AlexNet, InceptionV3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=1)
    
    parser.add_argument("-o", "--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("-m", "--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("-tr", "--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("-te", "--test", type=str, default=os.environ["SM_CHANNEL_TEST"])
    
    parser.add_argument("-s3", "--s3-bucket", type=str, default=os.environ["S3_BUCKET"])
    parser.add_argument("--model-name", type=str, default=os.environ["MODEL_NAME"])
    parser.add_argument("--fine-tune", type=bool, default=os.environ["FINE_TUNE"])
    parser.add_argument("--pretrained", type=bool, default=os.environ["PRETRAINED"])
    
    
    args = parser.parse_args()
    if args.model_name == "resnet18":
        model = CarModule(args, ResNet18(fine_tune=args.fine_tune, pretrained=args.pretrained))
    elif args.model_name == "resnet34":
        model = CarModule(args, ResNet34(fine_tune=args.fine_tune, pretrained=args.pretrained))
    elif args.model_name == "resnet50":
        model = CarModule(args, ResNet50(fine_tune=args.fine_tune, pretrained=args.pretrained))
    elif args.model_name == "resnet101":
        model = CarModule(args, ResNet101(fine_tune=args.fine_tune, pretrained=args.pretrained))
    elif args.model_name == "resnet152":
        model = CarModule(args, ResNet152(fine_tune=args.fine_tune, pretrained=args.pretrained))
    elif args.model_name == "vgg11":
        model = CarModule(args, VGG11(fine_tune=args.fine_tune, pretrained=args.pretrained))
    elif args.model_name == "vgg13":
        model = CarModule(args, VGG13(fine_tune=args.fine_tune, pretrained=args.pretrained))
    elif args.model_name == "vgg16":
        model = CarModule(args, VGG16(fine_tune=args.fine_tune, pretrained=args.pretrained))
    elif args.model_name == "vgg19":
        model = CarModule(args, VGG19(fine_tune=args.fine_tune, pretrained=args.pretrained))
    elif args.model_name == "alexnet":
        model = CarModule(args, AlexNet(fine_tune=args.fine_tune, pretrained=args.pretrained))
    elif args.model_name == "inceptionv3":
        model = CarModule(args, InceptionV3(fine_tune=args.fine_tune, pretrained=args.pretrained))
    else:
        raise ValueError("Invalid model name")

    tensorboard_logger = TensorBoardLogger(
        save_dir=args.output_data_dir,
        name="lightning_logs",
        version=0,
    )
    
    trainer = Trainer(gpus=args.gpus, max_epochs=args.epochs, default_root_dir=args.output_data_dir, logger=tensorboard_logger)
    trainer.fit(model)
    with open(os.path.join(args.model_dir, "model.pth"), "wb") as f:
        torch.save(model.model.state_dict(), f)
