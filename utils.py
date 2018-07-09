from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
# define global variables
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
# preprocessing data
# create data loader with transformations attached
train_transforms = transforms.Compose([transforms.RandomVerticalFlip(),
                                       transforms.Resize((256, 256)),
                                       transforms.RandomCrop((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean, std)])

validate_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                          transforms.CenterCrop((224,
                                                                 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])

test_transforms = validate_transforms

transform_map = {
    'train': train_transforms,
    'valid': validate_transforms,
    'test': test_transforms
}


def preprocessData(image_dir='', data_type='train'):
    """this function create dataloader for train, validate, test data located
    in image_dir

    :image_dir: data path where images are located
    :data_type: 'train', 'validate', or 'test'
    :return: dataloader

    """
    if data_type not in transform_map.keys():
        return None
    # directories for train, validate, and test data
    data_path = image_dir + data_type
    # define transforms
    # generate data loaders
    datasets = ImageFolder(data_path, transform=transform_map[data_type])
    if 'train' == data_type:
        dataloader = DataLoader(datasets, batch_size=64, shuffle=True)
        pass
    else:
        dataloader = DataLoader(datasets, batch_size=64)
        pass
    # return
    return dataloader, datasets.class_to_idx
