import os
import itertools

from studio.utils import utils
from studio.settings.dgx import DGX
from studio.steps.train import Train


class Tuner(object):

    def __init__(self, config_yaml):
        self.config = utils.read_yaml(config_yaml)
        self.schema = utils.read_schema('tuner')
        utils.validate_config(self.config, 'tuner', defaults=True)
        self.tunables = self.process_tunables()

        # Get the same time stamp for all experiment folders
        self.date = utils.timestamp().replace('/', '_')

    def process_tunables(self):
        """
        Maps tunables to their values specified in the tuner .yaml file
        Only the following tunables are supported: `lr`, `batch_size`
        """
        # Find the parameters in the schema where `tunable` is true
        tunable_names = utils.traverse(self.schema, ' ', [], 'tunable')

        # Find the values of each tunable
        tunable_values = {}
        for tunable in tunable_names:
            if tunable == 'lr':
                tunable_values[tunable] = list(map(str, self.config['steps']['train']['optimizer']['SGD']['lr']))
            elif tunable == 'batch_size':
                tunable_values[tunable] = list(map(str, self.config['steps']['train']['hyperparameters']['batch_size']))
        return tunable_values

    def _identifier(self):
        """
        Maps config parameters into a single string that shortly
        summarizes the content of config's fields. Fields a sorted
        to provide deterministic output.

        Currently it only attaches:
            - date
            - experiment author
            - experiment name
        """
        experiment_name = self.config['experiment']['name']
        experiment_author = self.config['experiment']['author'].split('@')[0]
        date = self.date.replace(':', '-')

        identifier = '_'.join([date, experiment_author, experiment_name])
        return identifier

    def process_experiment(self, config_experiment, combination):
        self.user = config_experiment['author']
        self.experiment_id = self._identifier()
        self.output_dir = config_experiment['output_dir']
        self.experiment_dir = os.path.join(self.output_dir, self.experiment_id, '/'.join(list(combination)))

        # create local folders
        utils.mkdir(self.experiment_dir)

    def process_settings(self, config_settings):
        for setting in config_settings:
            if setting == 'dgx':
                # set up DGX settings
                num_gpus = config_settings['dgx']['num_gpus']
                max_gpus = config_settings['dgx']['max_gpus']

                self.dgx = DGX()
                self.dgx.allocate_GPUs(num_gpus=num_gpus, max_GPUs=max_gpus)

            # elif setting == 'lab':
                # set up Lab settings
            #    self.lab = Lab(lab_API_key)
            #    self.lab.auth()

    def process_steps(self, config_steps):
        for step in config_steps.keys():
            # create local folders
            step_folder = os.path.join(self.experiment_dir, step)
            utils.mkdir(step_folder)

            # process train step
            if step == 'train':
                self.train = Train(dgx=self.dgx,
                                   # lab=self.lab,
                                   config=config_steps['train'],
                                   output_dir=step_folder)
                self.train.run()
                config_steps['train'].update(self.train.config)

            # process eval step
            # elif step == 'eval':
            #    self.eval = Eval(data=self.data,
            #                     eval_config=config_steps['eval'],
            #                     output_dir=step_folder)
            #    self.eval.run()
            #    config_steps['eval'].update(self.eval.config)

    def run_studio(self, config, combination):
        self.process_experiment(config['experiment'], combination)
        self.process_settings(config['settings'])
        self.process_steps(config['steps'])

        experiment_config = os.path.join(self.experiment_dir, 'experiment_config.json')
        utils.store_json(config, experiment_config)

    def run(self):
        # Combine each tunable value with the tunable's name for experiments folder names
        tunable_named_values = [list(map(lambda x:"_".join([tunable_name, x]), tunable_values))
                                for tunable_name, tunable_values in self.tunables.items()]
        # Compute all the tuning combinations
        tuning_combinations = list(itertools.product(*list(tunable_named_values)))

        for combination in tuning_combinations:
            # Overright the config with the combination values
            for tunable in combination:
                tunable_name, tunable_value = tunable.rsplit('_', 1)
                if tunable_name == 'lr':
                    self.config['steps']['train']['optimizer']['SGD']['lr'] = float(tunable_value)
                elif tunable_name == 'batch_size':
                    self.config['steps']['train']['hyperparameters']['batch_size'] = int(tunable_value)
            # Run studio experiment for the combination
            self.run_studio(self.config, combination)
