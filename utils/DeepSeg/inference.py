from utils.DeepSeg.config import *
import numpy as np
from cv2 import imread, resize, INTER_NEAREST
from keras import backend as K
# from utils.DeepSeg.models import get_deepseg_model
from pathlib import Path
import tensorflow as tf

K.set_image_data_format('channels_last')
if K.image_data_format() == 'channels_first':
    IMAGE_ORDERING = 'channels_first'
elif K.image_data_format() == 'channels_last':
    IMAGE_ORDERING = 'channels_last'

class TumorSegmentor():
    def __init__(self, model_path = "./models/deep-seg/densenet121_unet_mod.keras"):
        if Path(model_path).exists():
            self.model = tf.keras.models.load_model(model_path, compile=False)
        else:
            from utils.DeepSeg.models import get_deepseg_model
            # create the DeepSeg model
            self.model = get_deepseg_model(
                encoder_name=config['encoder_name'], 
                decoder_name=config['decoder_name'], 
                n_classes=config['n_classes'], 
                input_height=config['input_height'], 
                input_width=config['input_width'], 
                depth=config['model_depth'], 
                filter_size=config['filter_size'], 
                up_layer=config['up_layer'],
                trainable=config['trainable'], 
                load_model=config['load_model'])

    def preprocess(self, img, width, height, imgNorm="norm", odering='channels_first'):
        # img = load_h5(img_path)[0][:, :, 2]
        img = ((img-img.min())/(img.max()-img.min()))*255 # scale to be 0 to 255 (uint8)
        img = img.astype("uint8")
        img = np.dstack([img, img, img])
        if imgNorm == "sub_and_divide":
            img = np.float32(resize(img, (width, height), interpolation = INTER_NEAREST)) / 127.5 - 1
        elif imgNorm == "sub_mean":
            img = resize(img, (width, height))
            img = img.astype(np.float32)
            img[:,:,0] -= 103.939
            img[:,:,1] -= 116.779
            img[:,:,2] -= 123.68
            img = img[ :, :, ::-1 ]
        elif imgNorm == "divide":
            img = resize(img, (width, height), interpolation = INTER_NEAREST)
            img = img.astype(np.float32)
            img = img/255.0
        elif imgNorm == "norm":
            # Intensity normalization (zero mean and unit variance)
            img = resize(img, (width, height), interpolation = INTER_NEAREST)
            img_mean = img.mean()
            img_std = img.std()
            # img = (img - img_mean) / img_std
            if(img_std != 0):
                img = (img - img_mean) / img_std
            else:
                # print('Error!!: invalid value encountered in true_divide')
                img = (img - img_mean)

        if odering == 'channels_first':
            img = np.rollaxis(img, 2, 0)
        return img
    
    def predict(self, inp=None):
        output_width = config['output_width']
        output_height  = config['output_height']
        input_width = config['input_width']
        input_height = config['input_height']
        n_classes = config['n_classes']
    
        arr = self.preprocess(inp, input_width, input_height, odering=IMAGE_ORDERING)
        pr = self.model.predict(np.array([arr]), verbose=False)[0] # (50176, 2)
        # comapare the two channels and get the max value (with 1 in new array)
        # pr = pr.reshape((output_height,  output_width, n_classes)).argmax(axis=2) # (224, 224)
        pr = pr.reshape((output_height,  output_width, n_classes)) # (224, 224)
        # print(pr)
        # # change the predicted label 3 back to value of 4 (standard BraTS labels)
        # pr[pr==3] = 4
        return pr[:, :, 1]
