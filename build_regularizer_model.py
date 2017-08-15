from keras.layers import Input, Embedding, Flatten, Dense, Dropout, Concatenate
from keras.model import Model
from covariance_regularizer import *


def build_regularizer_model(users,
                            movies,
                            corr_weight=1,
                            mean_weight=1,
                            ignore_variance=False,
                            user_dims=50,
                            movie_dims=50,
                            dense=[100],
                            layer_reg=None,
                            dropout=[0.5],
                            opt='Nadm',
                            act='ELU'
                            loss='mse', ):
    """Builds a model for generating movie ratings
    uses a covariance regularizer to force linear independence of embeddings
    input parameters allow for more fine control"""
    user_in = Input(shape=(1,), dtype='int64', name='user_in')
    u = Embedding(users,
                  50,
                  embeddings_regularizer=CovarianceRegularizer(1, 1),
                  name='user_embedding',)(user_in)
    movie_in = Input(shape=(1,), dtype='int64', name='movie_in')
    m = Embedding(movies,
                  50,
                  embeddings_regularizer=CovarianceRegularizer(1, 1),
                  name='movie_embedding',)(movie_in)
    x = Concatenate([u, m])
    x = Flatten(x)
    if len(dropout) % len(dense) == 0:
        dropout = np.tile(dropout, len(dropout) // len(dense))
    else:
        print("Number of dense layers not multiple of number of dropout layers. Repeating first droupout value")
        dropout = np.tile(dropout[0], len(dense))
    for nodes, drop in zip(dense, dropout):
        x = Dense(nodes, activation=act, kernel_regularizer=layer_reg)(x)
        x = Dropout(drop)(x)
    x = Dense(1, name='rating_output')(x)

    regularizer_model = Model([user_in, movie_in], x, )
    regularizer_model.compile(opt, loss=loss, )
    return(regularizer_model)
