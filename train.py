# import libraries
from utils import preprocessData
from model_funcs import createNetWork, performTraining, saveCheckPoint


# main
if __name__ == "__main__":
    arch = 'vgg19_bn'
    hidden_units = [4096, 4096, 1024]
    # preprocess data
    train_loader, class_to_idx = preprocessData("./flowers/", "train")
    validate_loader, _ = preprocessData("./flowers/", "valid")
    # create network
    model, criterion, optimizer = \
        createNetWork(arch, 0.001, hidden_units)
    # train
    performTraining(model, train_loader, validate_loader, optimizer, criterion)
    print("training process is finished")
    # save checkpoint
    save_path = "checkpoint.pth"
    saveCheckPoint(model, arch, hidden_units,
                   class_to_idx,
                   save_path)
    print("check point is saved to " + save_path)
    pass
