from tensorflow.keras.callbacks import EarlyStopping, History, ModelCheckpoint
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from math import pi, log
from .model_types import mvn_predictor
import numpy as np
from scipy.stats import multivariate_normal
import os

num_threads = int(os.environ.get("num_threads", 2))
tf.config.threading.set_inter_op_parallelism_threads(num_threads)


@tf.function
def layer_to_mean_covinv(final_layer, k):
    mean = final_layer[:, :k]
    ### Means is now a nx2 matrix
    cov = final_layer[:, k:]
    Ltri = tfp.math.fill_triangular(cov)

    b = tfp.bijectors.TransformDiagonal(tfp.bijectors.Exp())
    Ltri = b.forward(Ltri)
    perm_transpose = [0, 2, 1]
    cov_inv = tf.matmul(Ltri, tf.transpose(Ltri, perm=perm_transpose))
    return mean, Ltri, cov_inv


class nll_loss:
    def __init__(self, k, dtype="float32"):
        self.k = k
        self.mvn_const = tf.constant(self.k / 2 * log(2 * pi), dtype=dtype)
        self.__name__ = "nll_loss"

    @tf.function
    def __call__(self, Y, final_layer):
        ### Mean
        mean, Ltri, cov_inv = layer_to_mean_covinv(final_layer, self.k)
        N = tf.cast(tf.shape(Y)[0], tf.float32)
        diff = Y - mean
        diff = tf.expand_dims(diff, 2)
        perm_transpose = [0, 2, 1]

        diff_transpose = tf.transpose(diff, perm=perm_transpose)
        p3_covariance = diff_transpose @ cov_inv
        p3_covariance = p3_covariance @ diff
        diag = tf.math.log(tf.linalg.diag_part(Ltri))
        log_det = tf.math.reduce_sum(diag, axis=1)
        p3_covariance = tf.reshape(p3_covariance, (-1,))
        negll_comp = 0.5 * p3_covariance - log_det + self.mvn_const
        negll_comp = tf.reshape(negll_comp, (-1,))
        return negll_comp


def build_model(
    N_input,
    loss_fn,
    N_output=5,
    hidden_layers=[100, 50],
    act=tf.nn.relu,
    lr=0.01,
    lr_decay=False,
    modelname=None,
):
    inputs = tf.keras.Input(shape=(N_input,))
    x = inputs
    for k in hidden_layers:
        x = tf.keras.layers.Dense(k)(x)
        x = act(x)
    outputs = tf.keras.layers.Dense(N_output)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    if lr_decay:
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr, decay_steps=10000, decay_rate=0.95
        )
    else:
        lr_schedule = lr
    # optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
    opt = tf.keras.optimizers.Adam(lr=lr_schedule)
    model.compile(loss=loss_fn, optimizer=opt)
    return model


class mvn_neuralnetwork(mvn_predictor):
    def __init__(
        self,
        loss_kwargs={},
        hidden_layers=[20, 20],
        fname="",
        learning_rate=0.01,
        lr_decay=False,
        modelname=None,
    ):
        self.loss_kwargs = loss_kwargs
        self.hidden_layers = hidden_layers
        self.fname = fname
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay


    def fit(
        self,
        X_train,
        Y_train,
        X_valid,
        Y_valid,
        cp_name="",
        retrain=False,
        epochs=1000,
        batch_size=256,
        **kwargs,
    ):
        p = Y_train.shape[1]
        self.p=p
        N_output = int(p + p*(p+1)/2)
        es = EarlyStopping(monitor="val_loss", patience=50)
        loss = nll_loss(p, **self.loss_kwargs)
        self.model = build_model(
            X_train.shape[1],
            loss_fn=loss,
            hidden_layers=self.hidden_layers,
            lr=self.learning_rate,
            lr_decay=self.lr_decay,
            N_output=N_output
        )
        self.hist = History()

        # Helps stop the conflicting model names
        f_name_append = cp_name + self.fname

        cbs = [es, self.hist]

        if not retrain:
            cp_file = f"best_model{f_name_append}.h5"
            mc = ModelCheckpoint(
                filepath=cp_file,
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
                save_best_only=True,
            )
            cbs.append(mc)
        self.model.fit(
            X_train,
            Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_valid, Y_valid),
            callbacks=cbs,
            **kwargs,
        )

        if retrain:
            self.model = build_model(
                X_train.shape[1],
                loss_fn=loss,
                hidden_layers=self.hidden_layers,
                lr=self.learning_rate,
                lr_decay=self.lr_decay,
                N_output=N_output
            )
            X = np.vstack([X_train, X_valid])
            Y = np.vstack([Y_train, Y_valid])
            best_val_loss = np.argmin(self.hist.history["val_loss"])
            print("#" * 10)
            print(best_val_loss)

            self.model.fit(
                X,
                Y,
                epochs=best_val_loss,
                batch_size=batch_size,
                verbose=1,
            )
        else:
            self.model.load_weights(cp_file)
            os.remove(cp_file)

    def scipy_distribution(self, X, cmat_output=False):
        preds = self.model.predict(X)
        print(preds.shape)
        mean, _, cov_inv = layer_to_mean_covinv(preds, self.p)
        mean = mean.numpy().copy()
        cov_inv = cov_inv.numpy().copy()
        cov = np.linalg.inv(cov_inv)
        N = X.shape[0]
        if cmat_output:
            out = [mean, cov]
        else:
            out = [multivariate_normal(mean[i, :], cov=cov[i, :, :]) for i in range(N)]
        return out
