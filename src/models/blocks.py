import tensorflow as tf
from omegaconf.dictconfig import DictConfig
tfk = tf.keras
tfkl = tfk.layers
K = tfk.backend

from src.utils import f1

class DenseNet121():
    def __init__(self, config: DictConfig):
        self.config = config

    def get_compiled_model(self, class_weights = None) -> tfk.Model:
        mb_conf = self.config.model_builder
        input_shape = mb_conf.input_shape
        weights = mb_conf.weights
        activation = mb_conf.activation
        n_classes = mb_conf.n_classes
        input = tfkl.Input(shape=input_shape)
        base_model = tfk.applications.densenet.DenseNet121(
            include_top=False,
            input_tensor=input,
            input_shape=input_shape,
            weights=weights,
            pooling="avg",
        )
        x = base_model.output
        x = tfk.layers.Dropout(0.15)(x)

        prediction = tfkl.Dense(n_classes, activation=activation, name='classifier')(x)
        model = tfk.Model(inputs=input, outputs=prediction)

        # compiling
        optimizer = tfk.optimizers.legacy.Adam(learning_rate=2.5e-6)
        metrics = [
            tfk.metrics.SensitivityAtSpecificity(0.8),
            tfk.metrics.AUC(curve="PR", name="AUC of Precision-Recall Curve"),
            tfk.metrics.FalseNegatives(),
            tfk.metrics.FalsePositives(),
            tfk.metrics.TrueNegatives(),
            tfk.metrics.TruePositives(),
            f1
        ]

        if class_weights is not None:
            loss = self.__get_weighted_loss(weights=class_weights)
        else:
            loss = K.binary_crossentropy
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model
    
    @staticmethod
    def __get_weighted_loss(weights):
        def weighted_loss(y_true, y_pred):
            loss_output = K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
            return loss_output
        return weighted_loss

class EfficientNetB2():
    def __init__(self, config: DictConfig):
        self.config = config

    def get_compiled_model(self, class_weights = None) -> tfk.Model:
        mb_conf = self.config.model_builder
        input_shape = mb_conf.input_shape
        weights = mb_conf.weights
        activation = mb_conf.activation
        n_classes = mb_conf.n_classes
        input = tfkl.Input(shape=input_shape)
        base_model = tfk.applications.efficientnet.EfficientNetB2(
            include_top=False,
            input_tensor=input,
            input_shape=input_shape,
            weights=weights,
            pooling="avg",
        )
        x = base_model.output
        x = tfk.layers.Dropout(0.15)(x)

        prediction = tfkl.Dense(n_classes, activation=activation, name='classifier')(x)
        model = tfk.Model(inputs=input, outputs=prediction)

        # compiling
        optimizer = tfk.optimizers.AdamW(learning_rate=1e-6)
        metrics = [
            tfk.metrics.SensitivityAtSpecificity(0.8),
            tfk.metrics.AUC(curve="PR", name="AUC of Precision-Recall Curve"),
            tfk.metrics.FalseNegatives(),
            tfk.metrics.FalsePositives(),
            tfk.metrics.TrueNegatives(),
            tfk.metrics.TruePositives(),
            f1
        ]

        if class_weights is not None:
            loss = self.__get_weighted_loss(weights=class_weights)
        else:
            loss = K.binary_crossentropy
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model
    
    @staticmethod
    def __get_weighted_loss(weights):
        def weighted_loss(y_true, y_pred):
            loss_output = K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
            return loss_output
        return weighted_loss
