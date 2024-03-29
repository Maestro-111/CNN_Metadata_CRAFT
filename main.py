from data_prep import save_to_dataset
from data_prep import delete_files_in_directory
from data_prep import convert_to_gray
from data_prep import augementation
from data_prep import sharp_and_res
from data_prep import check_shape


from custom_CNN import create_dataset
from custom_CNN import make_confusion_matrix
from custom_CNN import neural_net_mixin
from custom_CNN import CNN

import numpy as np
import tensorflow as tf
from tensorflow import keras

#'C:\metadata_craft'

def process_data(source:str,aug:bool,factor=6):

    save_to_dataset(data_dir=source)
    sharp_and_res(data_dir = "dataset",factor=factor)

    if aug:
        augementation('dataset/train/key_plates', 1100, 'surveys')
        augementation('dataset/test/key_plates', 100, 'surveys')
        augementation('dataset/validation/key_plates', 300, 'surveys')

        augementation('dataset/train/other', 1100, 'surveys')
        augementation('dataset/test/other', 100, 'surveys')
        augementation('dataset/validation/other', 300, 'surveys')





def dataset(train_path,val_path,test_path,dim:tuple,color_mode:str):
    train_dataset, validation_dataset, test_dataset, class_names, num_classes = create_dataset(train_path,val_path,test_path,dim,color_mode)

    print(class_names)
    print(num_classes)

    return train_dataset, validation_dataset, test_dataset, class_names, num_classes


def delete_data(classes=['key_plates', 'other']):
    for type_dir in ['train', 'test', 'validation']:
        for type_image in classes:
            path = f'dataset/{type_dir}/{type_image}'
            delete_files_in_directory(directory_path=path)


def pipeline(delete=False,process=False,aug=False,train=False,dim=tuple([]),source='C:\metadata_craft'):

    if delete:
        delete_data()
    if process:
        process_data(source=source,aug=aug)

    if train:

        if not dim:
            raise ValueError

        width,height,length = dim

        if length == 1:
            color_mode = 'grayscale'
        else: # 3
            color_mode = 'rgb'

        train_dataset, validation_dataset, test_dataset, class_names, num_classes = dataset(
            "dataset/train",
            "dataset/validation",
            "dataset/test",
            (width,height),
            color_mode)


        train_model_and_save(train_dataset, validation_dataset, test_dataset, class_names, num_classes, dim)


def train_model_and_save(train_dataset, validation_dataset, test_dataset,class_names, num_classes, dim):

    """
    train and eval CNN net

    """


    CNN_net = CNN(num_classes, dim)
    history = CNN_net.train(train_dataset,validation_dataset,epochs=45)
    CNN_net.summary()
    CNN_net.plot_training_hist(history, '3-layers CNN', ['red', 'orange'], ['blue', 'green'])
    CNN_net.evaluate_model(test_dataset,class_names)
    CNN_net.save('model.h5')


pipeline(delete=False,process=False,aug=False,train=True,dim=(224,224,3),source='C:\metadata_craft') # 224*224, image is rgb