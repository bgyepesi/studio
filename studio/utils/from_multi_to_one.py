import os
import sys
import keras
import tensorflow as tf

from keras.models import load_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ''


def convert(model_path):
    model = load_model(model_path, custom_objects={"tf": tf})
    return next((layer for layer in model.layers if isinstance(layer, keras.engine.training.Model)), None)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: from_multi_to_one_gpu.py <model_path> <model_1_gpu_name>')
        sys.exit(1)
    print('Converting model: ', sys.argv[1])
    model_path = sys.argv[1]
    model_stripped = convert(model_path)
    model_stripped.save(os.path.join(os.path.dirname(model_path), sys.argv[2]))
    print('New model saved as ', os.path.join(os.path.dirname(model_path), sys.argv[2]))
