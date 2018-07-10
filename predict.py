# import libraries
from model_funcs import loadCheckPoint, predictWithModel
import json
import argparse
# parse input arguments


def parseInputsForPredicting():
    """this function parse input arguments for predicting
    :returns: result

    """
    parser = argparse.ArgumentParser(
        description='predicting with trained model')
    parser.add_argument("input", action='store',
                        help='path to input image')
    parser.add_argument("checkpoint", action='store',
                        help='path to input checkpoint')
    parser.add_argument("--top_k", action='store', type=int,
                        default=3, dest='top_k',
                        help='top_k default is 3')
    parser.add_argument("--category_names", action='store',
                        default="", dest='category_names',
                        help='path to cat_to_name.json or similar')
    parser.add_argument("--gpu", action='store_true', default=False,
                        dest='use_gpu',
                        help='use gpu or not, False by default')
    return parser.parse_args()


# main
if __name__ == "__main__":
    # parse input arguments
    results = parseInputsForPredicting()
    print("input is {!r}".format(results.input))
    print("checkpoint is {!r}".format(results.checkpoint))
    print("top_k is {!r}".format(results.top_k))
    print("category_names is {!r}".format(results.category_names))
    print("use_gpu is {!r}".format(results.use_gpu))
    # load check point to model
    model = loadCheckPoint(results.checkpoint)
    # predict
    if results.category_names:
        with open(results.category_names, 'r') as f:
            cat_to_name = json.load(f)
            pass
        pass
    else:
        cat_to_name = ''
        pass
    image_path = results.input
    topk = results.top_k
    use_gpu = results.use_gpu
    probs, classes = predictWithModel(image_path, model, topk=topk,
                                      cat_to_name=cat_to_name,
                                      use_gpu=use_gpu)
    print("top{} classes: ".format(topk), classes)
    print("top{} probs: ".format(topk), probs)
    pass
