import os
import tensorflow as tf
tfk = tf.keras

import numpy as np
import cv2
import matplotlib.pyplot as plt

class Explainer():
    def __init__(self, config, model) -> None:
        self.config = config
        self.model = model

        self.target_layer = self.config.interpretation.target_layer
        self.core_model = self.config.interpretation.core_model
        self.prediction_layer = self.config.interpretation.prediction_layer
        self.max_limit = self.config.interpretation.max_limit
        self.target_class_idx = self.config.interpretation.target_class_idx
        self.to_save = self.config.interpretation.to_save
        self.method_name = self.config.interpretation.method_name

        # get the core model if there is one 
        if self.core_model:
            self.cnn_model = self.model.get_layer(self.core_model)
        else:
            self.cnn_model = self.model
            
        # get the prediction layer as the output model
        if self.prediction_layer:
            output_layer = self.model.get_layer(self.prediction_layer).output
            self.cnn_model = tf.keras.Model(self.cnn_model.input, output_layer)

        # get the target layer as the attention layer
        if not self.target_layer:
            # get the last conv layer for explanation
            layer_names = [layer.name for layer in self.cnn_model.layers]
            for i in range(len(layer_names)-1, 0, -1):
                if 'conv' in layer_names[i]:
                    last_conv_layer_name = layer_names[i]
                    break
            self.target_layer = last_conv_layer_name

        # create the model for the gradcams
        self.grad_model = tf.keras.models.Model([self.cnn_model.input],
                                            [self.cnn_model.get_layer(self.target_layer).output,
                                            self.cnn_model.output])

    # def get_grad_cam(last_conv_layer_output, pred_model):
    def get_grad_cam(self, im):
        pred_index = self.target_class_idx

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = self.grad_model(im)
            if pred_index is None:
                pred_index = tf.argmax(preds[:, pred_index])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the output class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

        heatmap = tf.image.resize(heatmap[..., tf.newaxis], [im.shape[1], im.shape[2]])
        # if emphasize:
        #     heatmap = sigmoid(heatmap, 50, thresh, 1)
        heatmap = tf.cast((255 * heatmap), tf.uint8)          
        return heatmap

    @staticmethod
    def sigmoid(x, a, b, c):
        return c / (1 + np.exp(-a * (x-b)))

    def superimpose(self, img_bgr, cam, thresh, hif=0.002, emphasize=False):
        """
        Superimposes a grad-cam heatmap onto an image for model interpretation and visualization.

        Args:
        image: (img_width x img_height x 3) numpy array
        grad-cam heatmap: (img_width x img_width) numpy array
        threshold: float
        emphasize: boolean

        Returns 
        uint8 numpy array with shape (img_height, img_width, 3)
        """

        heatmap = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
        if emphasize:
            heatmap = self.sigmoid(heatmap, 50, thresh, 1)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # hif = .8
        # superimposed_img_gray = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2GRAY)
            
        # superimposed_img = superimposed_img.astype(np.uint8)  # scale 0 to 255  
        heatmap = cv2.cvtColor(heatmap.astype(np.uint8), cv2.COLOR_BGR2RGB)
        superimposed_img = heatmap * hif + img_bgr
        
        return superimposed_img
    
    def explain(self, data_seq):
        eval_df = data_seq.get_dataset_df()
        x_path = eval_df['Path'].values
        y_gt = eval_df['NumLabels'].values

        n_images = int(min(self.max_limit, len(x_path)))
        n_rows = int(np.ceil(np.sqrt(n_images)))
        n_cols = int(np.ceil(np.sqrt(n_images)))

        fig, ax = plt.subplots(n_rows, n_cols, subplot_kw={'xticks': [], 'yticks': []})
        fig.set_size_inches((10, 5))
        fig.set_frameon(False)

        for i in range(n_images):
            img = data_seq.load_image(x_path[i])
            img_array = np.expand_dims(img, axis=0)
            exp_result = self.get_grad_cam(img_array)
            y_true = y_gt[i]
            y_pred = self.model.predict(img_array)[:, 1]
            axis = ax[i//n_cols][i%n_cols]
            axis.set_title(f'{int(y_true)}, {int(np.round(y_pred))}')
            axis.axis('off')
            axis.imshow(exp_result, cmap='viridis')
            axis.imshow(img, alpha=0.4)
        
        plt.tight_layout()
        if self.to_save:
            plt.savefig(os.path.join(self.config.data_root_dir, f'{self.method_name}.png'),
                        dpi=300, bbox_inches='tight')
        plt.show()