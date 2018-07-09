from model_funcs import loadCheckPoint, predictWithModel
import json
# main
if __name__ == "__main__":
    # load check point to model
    model = loadCheckPoint("./checkpoint.pth")
    # predict
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    image_path = "./flowers/test/17/image_03830.jpg"
    topk = 3
    use_gpu = True
    probs, classes = predictWithModel(image_path, model, topk=topk,
                                      cat_to_name=cat_to_name,
                                      use_gpu=use_gpu)
    print("top{} classes: ".format(topk), classes)
    print("top{} probs: ".format(topk), probs)
    pass
