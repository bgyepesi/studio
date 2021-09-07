import os
import numpy as np

from numpy.random import seed
from studio.utils import utils
from studio.steps.eval import Eval
import tensorflow as tf
#from tensorflow import set_random_seed
from studio.settings.dgx import DGX
from studio.steps.train import Train
from studio.steps.data import TrainData, EvalData
from studio.settings.gcloud import GStorage


class Auto(object):

    def __init__(self, config_yaml):
        self.config = utils.read_yaml(config_yaml)
        utils.validate_config(self.config, 'auto', defaults=True)

    @staticmethod
    def get_identifier(author, experiment):
        """
        Maps config parameters into a single string that shortly summarizes the content of config's fields.
        Fields a sorted to provide deterministic output.

        Args:
            author: string indicating the author's name
            experiment: string indicating the experiment's name

        Returns:
            Single string containing:
                - date
                - experiment author
                - experiment name
        """
        date = utils.timestamp().replace('/', '_')
        date = date.replace(':', '-')

        identifier = '_'.join([date, author, experiment])
        return identifier

    def process_experiment(self):
        author = self.config['experiment']['author']
        experiment_name = self.config['experiment']['name']
        self.experiment_id = self.get_identifier(author, experiment_name)
        self.output_dir = self.config['experiment']['output_dir']
        self.experiment_dir = os.path.join(self.output_dir, self.experiment_id)

        # Fix experiment seed
        seed_value = self.config['experiment']['seed']
        if isinstance(seed_value, int):
            seed(seed_value)
            #set_random_seed(seed_value)
            tf.random.set_seed(seed_value)
        else:
            # generate a random seed value between 0 and 2**32 -1 as specified by numpy documentation here:
            # https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.RandomState.html#numpy.random.RandomState
            random_seed = np.random.randint(2**32)
            seed(random_seed)
            tf.random.set_seed(random_seed)
            #set_random_seed(random_seed)
            self.config['experiment']['seed'] = random_seed

        # create local folders
        utils.mkdir(self.experiment_dir)

    def process_settings(self):
        for name, settings in zip(self.config['settings'], self.config['settings'].values()):
            if name == 'dgx':
                # set up DGX settings
                self.num_gpus = settings['num_gpus']

                self.dgx = DGX()
                self.dgx.allocate_GPUs(num_gpus=self.num_gpus,
                                       max_GPUs=settings['max_gpus'])

            if name == 'gstorage':
                # set up GStorage settings
                self.gstorage = GStorage(project=settings['project'],
                                         bucket=settings['bucket'])

    def process_steps(self):
        for step in self.config['steps'].keys():
            # process data step
            if step == 'data':
                # process train data
                if self.config['steps']['data'].get('train'):
                    data = TrainData(config=self.config['steps']['data']['train'],
                                     output_dir=self.experiment_dir)
                    train_class_manifest_path, val_class_manifest_path = data.run()

                # update data `input` if `train` step is requested
                if self.config['steps'].get('train'):
                    self.config['steps']['train']['data']['input']['train_class_manifest_path'] = train_class_manifest_path
                    self.config['steps']['train']['data']['input']['val_class_manifest_path'] = val_class_manifest_path

                # process eval data
                if self.config['steps']['data'].get('eval'):
                    data = EvalData(config=self.config['steps']['data']['eval'],
                                    output_dir=self.experiment_dir)
                    test_class_manifest_path = data.run()

                # update `input` data if `eval` step is requested
                if self.config['steps'].get('eval'):
                    self.config['steps']['eval']['data']['input']['test_class_manifest_path'] = test_class_manifest_path

            # process train step
            if step == 'train':
                self.train = Train(config=self.config['steps']['train'],
                                   output_dir=self.experiment_dir,
                                   num_gpus=self.num_gpus)
                self.train.run()
                self.config['steps']['train'].update(self.train.config)

                # update `model_path` if `eval` step is requested
                if self.config['steps'].get('eval'):
                    self.config['steps']['eval']['single']['model_path'] = self.train.model_path
            # process eval step
            elif step == 'eval':
                self.eval = Eval(config=self.config['steps']['eval'],
                                 output_dir=self.experiment_dir)
                self.eval.id = self.experiment_id
                self.eval.run()

    def run(self):
        self.process_experiment()
        self.process_settings()
        self.process_steps()

        experiment_config = os.path.join(self.experiment_dir, 'experiment_config.json')
        utils.store_json(self.config, experiment_config)

        # Store copy to GStorage
        if self.config['settings'].get('gstorage'):
            self.gstorage.upload_folder(self.experiment_dir,
                                        self.config['settings']['gstorage']['dst_path'])
