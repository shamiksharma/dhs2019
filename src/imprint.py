from tensorflow import keras
from time import time
from sklearn.preprocessing import normalize
import numpy as np

class WeightImprint(keras.callbacks.Callback):
    def __init__(self, x_train, y_train):
        self.x = x_train
        self.y = y_train

    def on_epoch_begin(self, epoch, logs=None):
        if self.policy(epoch):
            start = time()
            self.imprint(self.model, self.x, self.y)
            end = time()
            print("Imprinted weights", end-start, "s")

    def policy(self, epoch):
        if epoch == 1:
            return True

        if epoch % 3 == 0:
            return True

        return False

    @staticmethod
    def imprint(model, x_train, y_train):
        # create bottleneck network and embeddings
        n_classes = len(np.unique(y_train))
        embedder_layers = model.layers[0]
        embedder = keras.models.Model(inputs=embedder_layers.input,
                                      outputs=embedder_layers.output)

        embeddings = embedder.predict(x_train, verbose=0, batch_size=1024)

        # create a matrix of per class average embedding
        indices = [[] for i in range(n_classes)]
        imprintings = np.zeros((n_classes, embeddings.shape[1]))

        for i, c in enumerate(y_train):
            indices[c].append(i)

        for c, i_group in enumerate(indices):
            embeddings_subset = embeddings[i_group]
            average_embedding = np.average(embeddings_subset)
            imprintings[c] = average_embedding

        # replace weight matrix of dense layer with imprinted weights
        imprintings = normalize(imprintings)
        coefficients = imprintings.T
        model.layers[-1].set_weights([coefficients])