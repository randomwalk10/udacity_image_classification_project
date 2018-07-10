
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

# In[1]:


# Imports here
import torch
import torchvision
from torchvision import transforms


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[2]:


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# In[3]:


# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomVerticalFlip(),
                                      transforms.Resize((256, 256)),
                                      transforms.RandomCrop((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406),
                                                           (0.229, 0.224, 0.225))])

validation_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                            transforms.CenterCrop((224, 224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406),
                                                                 (0.229, 0.224, 0.225))])

test_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                      transforms.CenterCrop((224, 224)),
                                      transforms.ToTensor(),  
                                      transforms.Normalize((0.485, 0.456, 0.406),
                                                           (0.229, 0.224, 0.225))])

# TODO: Load the datasets with ImageFolder
train_datasets = torchvision.datasets.ImageFolder(train_dir,
                                                  transform=train_transforms)
validation_datasets = torchvision.datasets.ImageFolder(valid_dir,
                                                       transform=validation_transforms)
test_datasets = torchvision.datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_datasets, batch_size=32)
test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=32)


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[4]:


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. If you want to talk through it with someone, chat with your fellow students! You can also ask questions on the forums or join the instructors in office hours.
# 
# Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.

# In[5]:


# TODO: Build and train your network
# load a pre-trained network(with gradient turned off)
# add a customized classifer to the network(new added classifier has gradient on automatically)
# perform training and validation(each epoch contains training and validation phases)


# In[6]:


# load a pre-trained network(with gradient turned off)
from torchvision import models
from torch import nn

model = models.vgg19_bn(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
    pass

model


# In[7]:


# add a customized classifer to the network(new added classifier has gradient on automatically)
classifier = nn.Sequential(nn.Linear(25088, 4096),
                           nn.ReLU(), nn.Dropout(0.2),
                           nn.Linear(4096, 4096),
                           nn.ReLU(), nn.Dropout(0.2),
                           nn.Linear(4096, 1024),
                           nn.ReLU(), nn.Dropout(0.2),
                           nn.Linear(1024, 102),
                           nn.LogSoftmax(dim=1))

model.classifier = classifier
model.classifier


# In[8]:


# prepare training
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.classifier.parameters())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


# In[9]:


# function for validation
def validate(model, dataloader, device, loss_func):
    # turn on eval mode
    model.eval()
    # turn off gradient tracking
    with torch.no_grad():
        loss_sum = 0.
        accuracy = 0.
        step = 0
        for inputs, labels in dataloader:
            # update step
            step += 1
            # load data to device
            inputs, labels = inputs.to(device), labels.to(device)
            # forward pass
            preds = model.forward(inputs)
            # loss
            loss = loss_func(preds, labels)
            loss_sum += loss
            # accuracy
            torch.exp_(preds)
            accuracy += (torch.max(preds, 1)[1]==labels).type(torch.FloatTensor).mean()
        pass
    loss_sum /= step
    accuracy /= step
    return loss_sum, accuracy


# In[10]:


# train and validate
epochs = 3
print_size = 10
for e in range(epochs):
    for ii, (inputs, labels) in enumerate(train_loader):
        # turn on train mode
        model.train()
        # load data to cuda
        inputs, labels = inputs.to(device), labels.to(device)
        # zero grad
        optimizer.zero_grad()
        # forward pass
        preds = model.forward(inputs)
        # loss
        loss = criterion(preds, labels)
        # backward prop
        loss.backward()
        # update optimizer
        optimizer.step()
        # perform validation every print_size iterations in training
        if (ii+1)%print_size == 0:
            loss_sum, accuracy = validate(model, validation_loader, device, criterion)
            print("epoch {}/{} validation at step {}: loss {}, accuracy {}".format(
                e+1, epochs, ii+1, loss_sum, accuracy))
        pass


# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[11]:


# TODO: Do validation on the test set
_, test_accuracy = validate(model, test_loader, device, criterion)
print("Results of validating on test set: accuracy {}".format(test_accuracy))


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[12]:


model_check_pt_path = "checkpoint.pth.tar"


# In[13]:


# TODO: Save the checkpoint 
check_point = {
    "epochs": epochs,
    "print_size": print_size,
    "state_dict": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "class_to_idx": train_datasets.class_to_idx,
}

torch.save(check_point, model_check_pt_path)


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[14]:


# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(pathname, model, optimizer):
    check_pt = torch.load(pathname)
    model.load_state_dict(check_pt["state_dict"])
    optimizer.load_state_dict(check_pt["optimizer"])
    model.class_to_idx = check_pt["class_to_idx"]
    return check_pt["epochs"], check_pt["print_size"]


# In[15]:


epochs, print_size = load_checkpoint(model_check_pt_path, model, optimizer)
print("loaded epochs {}, print_size {}".format(epochs, print_size))
_, test_accuracy = validate(model, test_loader, device, criterion)
print("Results of validating on test set: accuracy {}".format(test_accuracy))


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[16]:


from PIL import Image
import numpy as np
def process_image(im):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    # resize
    width, height = im.size
    if width>height:
        width *= 256./height
        width = int(width)
        height = 256
        pass
    else:
        height *= 256./width
        height = int(height)
        width = 256
        pass
    im = im.resize((width, height))
    # crop
    im = im.crop(((width-224)/2, (height-224)/2,
                    (width+224)/2, (height+224)/2))
    # convert to numpy array and do scale
    np_im = np.array(im)
    np_im = np_im / 255.0
    # normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    for ii, (u, d) in enumerate(zip(mean, std)):
        np_im[:,:,ii] = ( np_im[:,:,ii] - u ) / d
        pass
    # transpose
    np_im = np_im.transpose((2,0,1))
    # return
    return np_im
    
    


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[17]:


import matplotlib.pyplot as plt
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    if title:
        plt.title(title)
    ax.imshow(image)
    
    return ax


# In[28]:


target_class = '17'
image_path = "flowers/train/"+target_class+"/image_03833.jpg"
image = Image.open(image_path)
im = process_image(image)
imshow(im)


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[29]:


idx_to_class = {v: k for k, v in model.class_to_idx.items()}

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    # load image
    image = Image.open(image_path)
    im = process_image(image)
    # to tensor
    im = torch.from_numpy(im)
    # forward pass
    model.eval()
    with torch.no_grad():
        im.unsqueeze_(0)
        im = im.type(torch.FloatTensor)
        im = im.to(device)
        pred = model.forward(im)
        torch.exp_(pred)
        pass
    # get tok k
    pred.squeeze_(0)
    top_k_prob, top_k_idx = torch.topk(pred, topk)
    # return
    return [x.item() for x in top_k_prob], [idx_to_class[x.item()] for x in top_k_idx]


# In[30]:


topk = 5
probs, classes = predict(image_path, model, topk)
print(probs)
print(classes)


# In[31]:


print([cat_to_name[x] for x in classes])
print("target is ", cat_to_name[target_class])


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[32]:


# TODO: Display an image along with the top 5 classes
probs, classes = predict(image_path, model, topk)
# display origina image and its class name
im = Image.open(image_path)
im = process_image(im)
imshow(im, None, cat_to_name[target_class])
# display bar graph
y_loc = range(topk)
y_loc = y_loc[::-1]
plt.figure()
plt.barh(y_loc, probs, 0.5)
plt.yticks(y_loc, [cat_to_name[x] for x in classes])
plt.show()

    

