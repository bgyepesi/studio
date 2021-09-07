from keras import backend


def swish(x):
    """Swish activation function.

    # Arguments
        x: Input tensor.

    # Returns
        The Swish activation: `x * sigmoid(x)`.

    # References
        [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
    """
    if backend.backend() == 'tensorflow':
        try:
            # The native TF implementation has a more
            # memory-efficient gradient implementation
            return backend.tf.nn.swish(x)
        except AttributeError:
            pass

    return x * backend.sigmoid(x)
