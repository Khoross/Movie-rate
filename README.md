# Movie-rate
Movie ratings using embedding layers. Testing custom regularization methods

Goal for this repository is to produce a variety of user/movie rating model using an embedding matrix, and to investigate the resulting information contained within the embedding matricies.

Implemented models:

Models developed in this branch:
  Use a custom regularizer class to penalize the embedding matrix having nonzero mean, nonunit variance, and any nonzero covariance
    Provided the embedding matrix contains full dimensionality (and if it doesn't, this means the matrix has too many dimensions - not enough information is captured to use all of them), then this can be achieved by performing a single linear transformation.
    This moves the complexities from the embedding matrix into the weights/bias terms for the first layer.
    As such, I expect the regularizer loss to be very low/0
    This will make analyzing which of the vectors of the embedding matrix are most important difficult
    Doing this would require looking at the weights from the first layer in turn

Planned models:
  Use a custom regularizer to punish nonzero covariance.
    This can be achieved using a linear transformation, so loss should be low/0
    Allowing variance and mean to vary as they please will allow us to judge which are most important by comparing variance- higher variance means that more of the divergence is caused by this vector, provided the weights in the network are roughly equal
    Try this model both with and without an L2 regularizer on the dense layers
    In theory an L2 regularizer wil ensure that the weights are roughly evenly distributed, but at the same time, the variance of the embedding vectors should only change significantly if there is a reason to do so, so this shouldn't be too important
  Use an adversarial network to try and coerce the embedding matrix to being iid uniform(-1, 1)
    This is a much iffier prospect
    Not certain it's possible
    Pay careful attention to relative goodness of fit of this model vs others

Planned explorations:
  For a model with no regularizer, perform PCA on the embedding matricies and look at the movies at the top/bottom
  For models with regularizers, confirm very low covariance, and then do same (with and without PCA - compare hopefully low differences)
  Try and investigate model to determine importance of the different embedding vectors
  Ideally, figure out a way to attach a regularizer to this, to force all the variance into the embedding vectors
