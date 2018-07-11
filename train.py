# import libraries
from utils import preprocessData
from model_funcs import createNetWork, performTraining, saveCheckPoint
import argparse
import os

# parse input arguments


def parseInputsForTraining():
    """this function parse input arguments for training
    :returns: result

    """
    parser = argparse.ArgumentParser(description='train deep neural network')
    parser.add_argument("data_dir", action='store',
                        help='directory to image data')
    parser.add_argument("--save_dir", action='store',
                        default='./', dest='save_dir',
                        help='directory to save a checkpoint')
    parser.add_argument("--arch", action='store',
                        default='vgg11', dest='arch',
                        help='arch of pretrained model, '
                        'only vgg19_bn, resnet18, densenet121 are supported')
    parser.add_argument("--learning_rate", action='store', type=float,
                        default=0.001, dest='learning_rate',
                        help='learning rate, 0.001 by default')
    parser.add_argument("--hidden_units", action='store', nargs='+', type=int,
                        default=[], dest='hidden_units',
                        help='a list of units in hidden layers: int int int..')
    parser.add_argument("--epochs", action='store', type=int,
                        default=3, dest='epochs',
                        help='number of epochs for training, 3 by default')
    parser.add_argument("--gpu", action='store_true', default=False,
                        dest='use_gpu',
                        help='use gpu or not, False by default')
    return parser.parse_args()


# main
if __name__ == "__main__":
    results = parseInputsForTraining()
    print("data_dir is {!r}".format(results.data_dir))
    print("save_dir is {!r}".format(results.save_dir))
    print("arch is {!r}".format(results.arch))
    print("learning_rate is {!r}".format(results.learning_rate))
    print("hidden_units is {!r}".format(results.hidden_units))
    print("epochs is {!r}".format(results.epochs))
    print("use_gpu is {!r}".format(results.use_gpu))
    # preprocess data
    train_loader, class_to_idx = preprocessData(
        results.data_dir+os.path.sep, "train")
    validate_loader, _ = preprocessData(
        results.data_dir+os.path.sep, "valid")
    # create network
    model, criterion, optimizer = \
        createNetWork(results.arch, results.learning_rate,
                      results.hidden_units)
    # train
    performTraining(model, train_loader, validate_loader, optimizer, criterion,
                    epochs=results.epochs, use_gpu=results.use_gpu)
    print("training process is finished")
    # save checkpoint
    save_path = results.save_dir + os.path.sep + "checkpoint.pth"
    saveCheckPoint(model, results.arch, results.hidden_units,
                   class_to_idx, save_path)
    print("check point is saved to " + save_path)
    pass
