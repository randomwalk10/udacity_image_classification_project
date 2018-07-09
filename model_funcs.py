import torch
import torchvision
from torch.nn import Linear, Dropout, ReLU, LogSoftmax
from PIL import Image
from utils import test_transforms
# create a network combined with pretrained feature extractor and customized
# classifier


model_map = {
    'vgg11': torchvision.models.vgg11,
    'vgg11_bn': torchvision.models.vgg11_bn,
    'vgg13': torchvision.models.vgg13,
    'vgg13_bn': torchvision.models.vgg13_bn,
    'vgg16': torchvision.models.vgg16,
    'vgg16_bn': torchvision.models.vgg16_bn,
    'vgg19': torchvision.models.vgg19,
    'vgg19_bn': torchvision.models.vgg19_bn,
}


def createClassifier(hidden_units=[]):
    """this function create a classifier network

    :hidden_units: number of hidden units in classifier
    :returns: classifier

    """
    num_input = 512*7*7
    num_output = 102
    layers = []
    for i in range(len(hidden_units)):
        if not layers:
            layers.append(Linear(num_input, hidden_units[i]))
            pass
        else:
            layers.append(Linear(hidden_units[i-1], hidden_units[i]))
            pass
        layers.append(ReLU())
        layers.append(Dropout(0.2))
        pass
    if layers:
        layers.append(Linear(hidden_units[-1], num_output))
        pass
    else:
        layers.append(Linear(num_input, num_output))
        pass
    layers.append(LogSoftmax(dim=1))

    classifier = torch.nn.Sequential(*layers)
    return classifier


def createNetWork(arch, learn_rate=0.001, hidden_units=[]):
    """this function create a deep neural network for predicting flowers' name

    :arch: string name for pretrained network, e.g. "vgg13"
    :learn_rate: learning rate for optimizer ADAM
    :hidden_units: number of hidden units in classifier
    :returns: model, loss function, optimizer

    """
    # initialize pre-trained network
    if arch not in model_map.keys():
        return None, None, None
    model = model_map[arch](True)
    for params in model.parameters():
        params.requires_grad = False
        pass
    # initialize classifer
    model.classifier = createClassifier(hidden_units)
    # loss function
    criterion = torch.nn.NLLLoss()
    # initialize optimizer
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learn_rate)
    # return
    return model, criterion, optimizer

# perform training


def validateModel(model, data_loader, device, criterion):
    """ this function calculate accuracy and loss based on
    model and data loader
    """
    # turn on eval model
    model.eval()
    model.to(device)
    # calculate loss and accuracy
    loss_sum = 0.
    accuracy = 0.
    num_step = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            num_step += 1
            # prepare data
            inputs, labels = inputs.to(device), labels.to(device)
            # forward pass
            preds = model.forward(inputs)
            # loss
            loss = criterion(preds, labels)
            loss_sum += loss.item()
            # accuracy
            torch.exp_(preds)
            accuracy += (torch.max(preds, dim=1)[1] == labels).type(
                torch.FloatTensor).mean()
            pass
        accuracy /= num_step
        loss_sum /= num_step
    return accuracy, loss_sum


def performTraining(model, train_loader, validate_loader, optimizer, criterion,
                    epochs=3, validate_size=10, use_gpu=True):
    """TODO: Docstring for performTraining.

    :model: TODO
    :train_loader: TODO
    :validate_loader: TODO
    :optimizer: TODO
    :criterion: TODO
    :epochs: TODO
    :validate_size: TODO
    :device: TODO
    :returns: TODO

    """
    # prepare training
    train_loss_sum = 0.
    if torch.cuda.is_available() and use_gpu:
        device = torch.device('cuda')
        pass
    else:
        device = torch.device('cpu')
        pass
    model.to(device)
    # perform training
    for e in range(epochs):
        for ii, (inputs, labels) in enumerate(train_loader):
            # turn on training mode
            model.train()
            # prepare data
            inputs, labels = inputs.to(device), labels.to(device)
            # forward pass
            preds = model.forward(inputs)
            # loss
            train_loss = criterion(preds, labels)
            train_loss_sum += train_loss.item()
            # zero grad
            optimizer.zero_grad()
            # back prop
            train_loss.backward()
            # update optimizer
            optimizer.step()
            # perform validation
            if (ii+1) % validate_size == 0:
                # get loss and accuracy on test data set
                validate_accuracy, validate_loss = \
                    validateModel(model, validate_loader,
                                  device, criterion)
                # print out info
                print("epoch {}/{} at step {}: "
                      "training loss {}, "
                      "validate loss {}, "
                      "validate accuracy {}"
                      ".".format(e+1, epochs, ii+1,
                                 train_loss_sum / validate_size,
                                 validate_loss, validate_accuracy))
                # zero train_loss_sum
                train_loss_sum = 0.
                pass
            pass
        pass
    pass

# save check point


def saveCheckPoint(model, arch, hidden_units, class_to_idx,
                   save_dir="checkpoint.pth"):
    """this function save checkpoint of a trained network

    :model: model of a trained network
    :arch: arch of a pretrained network
    :hidden_units: layers of hidden network in the classifier
    :returns: NA

    """
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    checkpoint = {
        "arch": arch,
        "hidden_units": hidden_units,
        "idx_to_class": idx_to_class,
        "state_dict": model.state_dict()
    }
    torch.save(checkpoint, save_dir)
    pass

# load check point


def loadCheckPoint(load_dir):
    """this function load a checkpoint to model

    :load_dir: path to a checkpoint
    :returns: model of a trained network

    """
    check_pt = torch.load(load_dir)
    arch = check_pt["arch"]
    if arch not in model_map.keys():
        return None
    model = model_map[arch](True)
    model.classifier = createClassifier(check_pt["hidden_units"])
    for params in model.parameters():
        params.requires_grad = False
        pass
    model.load_state_dict(check_pt["state_dict"])
    model.idx_to_class = check_pt["idx_to_class"]
    return model

# predict


def getAccuracyWithModel(model, data_loader, use_gpu=True):
    # turn on eval model
    model.eval()
    if torch.cuda.is_available() and use_gpu:
        device = torch.device('cuda')
        pass
    else:
        device = torch.device('cpu')
        pass
    model.to(device)
    # get accuracy
    accuracy = 0.
    num_step = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            num_step += 1
            # prepare data
            inputs, labels = inputs.to(device), labels.to(device)
            # forward pass
            preds = model.forward(inputs)
            # update accuracy
            torch.exp_(preds)
            accuracy += (torch.max(preds, dim=1)[1] == labels).type(
                torch.FloatTensor).mean()
            pass
        accuracy /= num_step
        pass
    return accuracy


def predictWithModel(image_path, model, topk=3, cat_to_name={}, use_gpu=True):
    """this function make predications on topk probs and cats/names given
    input image path

    :image_path: path to image for prediction
    :model: trained network model
    :cat_to_name: dict mapping category to class name
    :use_gpu: use cuda or not
    :returns: return list of topk probabilities and categories/classes

    """
    # turn on eval model
    model.eval()
    if torch.cuda.is_available() and use_gpu:
        device = torch.device('cuda')
        pass
    else:
        device = torch.device('cpu')
        pass
    model.to(device)
    # make prediction
    with torch.no_grad():
        # load image
        im = Image.open(image_path)
        im = test_transforms(im)
        im.unsqueeze_(0)
        # prepare data
        im = im.to(device)
        # forward pass
        pred = model.forward(im)
        torch.exp_(pred)
        pred.squeeze_(0)
        pass
    # top k
    probs, top_idx = torch.topk(pred, topk)
    probs = [x.item() for x in probs]
    classes = [model.idx_to_class[x.item()] for x in top_idx]
    if cat_to_name:
        classes = [cat_to_name[x] for x in classes]
        pass
    # return
    return probs, classes
