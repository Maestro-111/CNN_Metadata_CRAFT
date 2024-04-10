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


TARGET_NAME = 'key_plates'
DIM = (224,224,3)
FACTOR = 7
SOURCE = 'C:\metadata_craft'
MODEL_NAME = 'model3'
EPOCHS = 12

def process_data(source:str,aug:bool,factor=6):

    save_to_dataset(data_dir=source)
    sharp_and_res(data_dir = "dataset",factor=factor)

    if aug:
        augementation(f'dataset/train/{TARGET_NAME}', 1700, 'surveys')
        augementation(f'dataset/validation/{TARGET_NAME}', 400, 'surveys')

    print("Data set has been created\n")


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



def pipeline(delete=False,process=False,aug=False,train_test=False):

    if delete:
        delete_data()
    if process:
        process_data(source=SOURCE,aug=aug,factor=FACTOR)

    if train_test:

        if not DIM or len(DIM) < 3:
            raise ValueError

        width,height,length = DIM

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


        train_test_model_and_save(train_dataset, validation_dataset, test_dataset, class_names, num_classes)


def train_test_model_and_save(train_dataset, validation_dataset, test_dataset,class_names, num_classes):

    """
    train and eval CNN net

    """

    CNN_net = CNN(num_classes, DIM, TARGET_NAME)
    history = CNN_net.train(train_dataset,validation_dataset,epochs=EPOCHS)
    CNN_net.summary()
    CNN_net.plot_training_hist(history, '3-layers CNN', ['red', 'orange'], ['blue', 'green'])
    CNN_net.evaluate_model(test_dataset,class_names)
    CNN_net.save(f'{MODEL_NAME}.h5')


pipeline(delete=True,process=True,aug=True,train_test=True)

