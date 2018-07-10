# example to run train.py

python train.py ./flowers/ --save_dir ./ --arch vgg19_bn --learning_rate 0.001 --hidden_units 4096 4096 1024 --epoch 4 --
gpu

# example to run predict.py

python predict.py ./flowers/test/17/image_03830.jpg ./checkpoint.pth --top_k 3 --category_names ./cat_to_name.json --gpu
