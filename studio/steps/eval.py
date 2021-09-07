import os

from PIL import ImageFile
from studio.utils import utils
from studio.evaluation.keras.evaluators import CNNEvaluator


# Solve High Resolution truncated files
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Eval(object):
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = os.path.join(output_dir, 'eval')
        if not os.path.isdir(self.output_dir):
            utils.mkdir(self.output_dir)

    def _save_results(self):
        self.evaluator.save_results(mode='average', id=self.evaluator.id, csv_path=self.output_dir, round_decimals=6)
        print('Average results saved in:', os.path.join(self.output_dir, self.evaluator.id + '_average.csv'))
        self.evaluator.save_results(mode='individual', id=self.evaluator.id, csv_path=self.output_dir, round_decimals=6)
        print('Individual results saved in:', os.path.join(self.output_dir, self.evaluator.id + '_individual.csv'))

    def run(self):
        self.evaluator = CNNEvaluator(id=self.id,
                                      model_path=self.config['single']['model_path'],
                                      ensemble_models_dir=self.config['single']['ensemble_models_dir'],
                                      verbose=self.config['single']['verbose'],
                                      concept_dictionary_path=self.config['single']['concept_dictionary_path'],
                                      batch_size=self.config['single']['batch_size']
                                      )

        self.evaluator.evaluate(data_dir="",  # This is a temporarly solution until `aip-eval` makes `data_dir` an optional argument
                                dataframe_path=self.config['data']['input']['test_class_manifest_path'],
                                top_k=self.config['single']['top_k'],
                                custom_crop=self.config['single']['custom_crop'],
                                data_augmentation=self.config['single']['data_augmentation'],
                                confusion_matrix=self.config['single']['confusion_matrix'],
                                save_confusion_matrix_path=os.path.join(self.output_dir, 'confusion_matrix.png'),
                                show_confusion_matrix_text=self.config['single']['show_confusion_matrix_text'],
                                validate_filenames=False
                                )

        self._save_results()
