# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

# train.py command
python train.py './flowers/' --gpu gpu --learning_rate 0.001 --epochs 6 --hidden_units 2048 --dropout 0.3 --epochs 6 --arch 'vgg16'
# predict.py command
python predict.py './flowers/test/1/image_06743.jpg' checkpoint.pth --topk 3 --category_names cat_to_name.json --gpu gpu

