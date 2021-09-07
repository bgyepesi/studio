import os
import keras
import shutil
import tensorflow


from studio.utils import utils
from studio.deployment.custom_layers import swish

# Disable al GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class Deploy:
    def __init__(self, config_yaml):
        self.config = utils.read_yaml(config_yaml)
        # utils.validate_config(self.config, 'deployment', defaults=True)
        self.config = self.config['deployment']

    @staticmethod
    def get_feature_layer(model, layer):
        if isinstance(layer, int):
            return model.layers[layer].output
        elif isinstance(layer, str):
            return model.get_layer(layer).output
        else:
            raise ValueError('Layer must be either a str or an int specifying the model layer, you inputed %s'
                             % str(type(layer)))

    @staticmethod
    def load_keras_model(model_path):
        custom_objects = {
            # Just in case you have Lambda layers which implicitly 'import tensorflow as tf'
            # (happens to be the case for some of our internal code)
            'tf': tensorflow,
            'os': os,
            # For mobilenets
            'relu6': keras.layers.ReLU(6, name='relu6'),
            'DepthwiseConv2D': keras.layers.DepthwiseConv2D,
            # For efficientnet
            'swish': swish
        }

        return keras.models.load_model(model_path, custom_objects=custom_objects)

    @staticmethod
    def convert_keras_to_tensorflow(model_path, output_dir, feature_layer=None, output_stripped_model_path=None):
        # Cut out to_multi_gpu stuff (this could possibly break some models which don't use to_multi_gpu)
        model = Deploy.load_keras_model(model_path)
        stripped_model = next((layer for layer in model.layers if isinstance(layer, keras.engine.training.Model)), None)
        if stripped_model:
            if output_stripped_model_path is None:
                output_stripped_model_path = '%s-stripped%s' % os.path.splitext(model_path)
            stripped_model.save(output_stripped_model_path)
            model_path = output_stripped_model_path

        keras.backend.clear_session()
        session = tensorflow.Session()
        keras.backend.set_session(session)

        # Disable loading of learning nodes
        keras.backend.set_learning_phase(0)
        model = Deploy.load_keras_model(model_path)

        builder = tensorflow.saved_model.builder.SavedModelBuilder(output_dir)

        signature_outputs = {
            'class_probabilities': model.output
        }

        if feature_layer is not None:
            signature_outputs.update({'image_features': Deploy.get_feature_layer(model, feature_layer)})

        signature = tensorflow.saved_model.signature_def_utils.predict_signature_def(
            inputs={
                'image': model.input
            },
            outputs=signature_outputs
        )

        builder.add_meta_graph_and_variables(
            sess=keras.backend.get_session(),
            tags=[tensorflow.saved_model.tag_constants.SERVING],
            signature_def_map={
                tensorflow.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
            }
        )
        builder.save()

    def run(self):
        if 'model_conversion' in self.config:
            conversion_cfg = self.config['model_conversion']
            input_model, output_dir = conversion_cfg['input_model'], conversion_cfg['output_dir']
            if not input_model.endswith('.h5') and not input_model.endswith('.hdf5'):
                raise ValueError('Input a correct model path')

            print('Converting Keras model {} to tensorflow \n'.format(input_model))
            if 'feature_layer' in conversion_cfg and isinstance(conversion_cfg['feature_layer'], int):
                self.convert_keras_to_tensorflow(input_model,
                                                 output_dir,
                                                 feature_layer=conversion_cfg['feature_layer'])
            else:
                self.convert_keras_to_tensorflow(input_model, output_dir)
            print('Keras model converted to tensorflow and saved in {}'.format(output_dir))
            # Zip model files
            shutil.make_archive('model', 'zip', output_dir)
            shutil.move('model.zip', os.path.join(output_dir, 'model.zip'))
            # Copy model_spec.json
            shutil.copy2(os.path.join('/'.join(input_model.split('/')[:-1]), 'model_spec.json'),
                         os.path.join(output_dir, 'model_spec.json'))
            # Remove created files
            os.remove(os.path.join(output_dir, 'saved_model.pb'))
            shutil.rmtree(os.path.join(output_dir, 'variables'))
            print('The model was converted and zipped successfully and it is ready for deployment.')
