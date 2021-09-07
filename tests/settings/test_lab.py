import re
import random
import string
import unittest
import studio.settings.lab as lab

from unittest.mock import patch
from dotmap import DotMap


class LabMock:
    # Define the config of the lab mock
    config = {'url': 'lab.aip.com',
              'api_key': '123456789'}

    # Define the html form to get the csrf token
    html_with_csrf_template = '''
<html>
<head>
<meta content="{}" name="csrf-token" />
<body>

</body>
</html>
'''
    # Define the html that will be filled with a specific token
    html_with_csrf = ""

    # Define the dictionary of the url patterns
    pattern_config = {}

    # Define the different regex supported
    # 1- Pattern for a word that contains only alphanumeric and _:
    #    used to define the name of datasets, snapshots, images and ontologies
    pattern_config['name_pattern'] = '[a-zA-Z0-9_]+'
    # 2- Pattern for base_url
    pattern_config['base_url_pattern'] = "^(https://|http://){0}(|/)$".format(
        config['url'])
    # 3- Pattern for the auth function
    pattern_config['dataset_pattern'] = pattern_config['base_url_pattern'].split(
        '(|/)')[0] + '/datasets(|/)$'
    # 4- Pattern for a specific dataset
    pattern_config['specific_dataset_pattern'] = pattern_config['dataset_pattern'].split(
        '(|/)')[0] + '/{0}(|/)$'.format(pattern_config['name_pattern'])
    # 5- Pattern for a specific image in a specific dataset
    pattern_config['specific_image_pattern'] = pattern_config['specific_dataset_pattern'].split(
        '(|/)')[0] + '/images/{0}(|/)$'.format(pattern_config['name_pattern'])
    # 6- Pattern for the post request during the snapshot creation
    pattern_config['snapshot_pattern'] = pattern_config['specific_dataset_pattern'].split(
        '(|/)')[0] + '/snapshots(|/)$'
    # 7- Pattern for a specific snapshot in a specific dataset
    pattern_config['specific_snapshot_pattern'] = pattern_config['specific_dataset_pattern'].split(
        '(|/)')[0] + '/snapshots/{0}(|/)$'.format(pattern_config['name_pattern'])
    # 8- Pattern for the get request during the snapshot creation
    pattern_config['new_snapshot_pattern'] = pattern_config['specific_dataset_pattern'].split(
        '(|/)')[0] + '/snapshots/new(|/)$'
    # 9- Pattern to get the manifest of a specific snapshot in a specific dataset
    pattern_config['specific_snapshot_manifest_pattern'] = pattern_config['specific_snapshot_pattern'].split(
        '(|/)')[0] + '/manifest(|/)$'
    # 10 - Pattern for a specific ontology
    pattern_config['specific_ontology_pattern'] = pattern_config['base_url_pattern'].split(
        '(|/)')[0] + '/ontology/{0}(|/)$'.format(pattern_config['name_pattern'])

    # Compile the regex patterns into a regex object
    regex_config = dict((key, re.compile(pattern))
                        for key, pattern in pattern_config.items())

    @staticmethod
    def check_headers(headers):
        return headers \
            and headers.get('Content-Type', None) == 'application/x-www-form-urlencoded; charset=UTF-8' \
            and headers.get('Connection', None) == 'keep-alive' \
            and headers.get('Authorization', None) == 'Token {0}'.format(LabMock.config['api_key'])

    @staticmethod
    def mocked_request_get(*args, **kwargs):
        """
        This function simulates the requests.get function by:
            - parsing the urls
            - dispatching the url to the appropriate function handler using the compiled regex object
            - pre-checking the authentication
        """
        # Check the authentication of the user
        url = args[0]
        if LabMock.config["url"] in url:
            headers = kwargs.get('headers', None)
            if LabMock.check_headers(headers):
                """
                The order of checking the url:
                    1. login
                    2. dataset exists
                    3. ontology exists
                    4. create new snapshot
                    5. snapshot exists
                    6. image exists
                    7. download manifest

                It is important to notice that the order of 4 and 5 cannot be swapped.
                This is because checking for 5 before 4 will lead to testing for a snapshot
                with name `new` instead of considering it as a request to create a new snapshot.
                """
                # 1. login
                if LabMock.regex_config['dataset_pattern'].match(url):
                    return DotMap({'text': 'Connected'})
                # 2. dataset exists
                elif LabMock.regex_config['specific_dataset_pattern'].match(url):
                    return LabMock.mocked_request_get_dataset_name_exists(*args, **kwargs)
                # 3. ontology exists
                elif LabMock.regex_config['specific_ontology_pattern'].match(url):
                    return LabMock.mocked_request_get_ontology_exists(*args, **kwargs)
                # 4. create new snapshot
                elif LabMock.regex_config['new_snapshot_pattern'].match(url):
                    return LabMock.mocked_request_get_create_snapshot(*args, **kwargs)
                # 5. snapshot exists
                elif LabMock.regex_config['specific_snapshot_pattern'].match(url):
                    return LabMock.mocked_request_get_dataset_snapshot_exists(*args, **kwargs)
                # 6. image exists
                elif LabMock.regex_config['specific_image_pattern'].match(url):
                    return LabMock.mocked_request_get_image_exists(*args, **kwargs)
                # 7. download manifest
                elif LabMock.regex_config['specific_snapshot_manifest_pattern'].match(url):
                    return LabMock.mocked_request_get_download_manifest(*args, **kwargs)

                return DotMap({'status_code': 404})

        # return a minimal login page
        return DotMap({'text': 'login.aip.com'})

    @staticmethod
    def filter_url(url):
        """ This function parses a given url `url` and returns its separate elements
        """
        list_url = url.replace('https://', '').replace('http://', '').split('/')
        return list(filter(('').__ne__, list_url))

    @staticmethod
    def mocked_request_get_dataset_name_exists(*args, **kwargs):
        list_url_parsed = LabMock.filter_url(args[0])
        if len(list_url_parsed) < 3:
            return DotMap({'status_code': 404})
        dataset_name = list_url_parsed[2]
        if dataset_name in LabMock.config['dataset']:
            return DotMap({'status_code': 200})
        return DotMap({'status_code': 404})

    @staticmethod
    def mocked_request_get_dataset_snapshot_exists(*args, **kwargs):
        dataset_name_result = LabMock.mocked_request_get_dataset_name_exists(
            *args, **kwargs)
        if dataset_name_result.status_code == 404:
            return dataset_name_result

        list_url_parsed = LabMock.filter_url(args[0])
        if len(list_url_parsed) < 5:
            return DotMap({'status_code': 404})
        dataset_name = list_url_parsed[2]
        snapshot_name = list_url_parsed[4]
        if snapshot_name in LabMock.config["snapshots"][dataset_name]:
            return DotMap({'status_code': 200})
        return DotMap({'status_code': 404})

    @staticmethod
    def mocked_request_get_ontology_exists(*args, **kwargs):
        list_url_parsed = LabMock.filter_url(args[0])
        if len(list_url_parsed) < 3:
            return DotMap({'status_code': 404})
        ontology_name = list_url_parsed[2]
        if ontology_name in LabMock.config['ontology']:
            return DotMap({'status_code': 200})
        return DotMap({'status_code': 404})

    @staticmethod
    def mocked_request_get_image_exists(*args, **kwargs):
        dataset_name_result = LabMock.mocked_request_get_dataset_name_exists(
            *args, **kwargs)
        if dataset_name_result.status_code == 404:
            return dataset_name_result

        list_url_parsed = LabMock.filter_url(args[0])
        if len(list_url_parsed) < 5:
            return DotMap({'status_code': 404})
        dataset_name = list_url_parsed[2]
        image_name = list_url_parsed[4]
        if image_name in LabMock.config["images"][dataset_name]:
            return DotMap({'status_code': 200})
        return DotMap({'status_code': 404})

    @staticmethod
    def mocked_request_get_download_manifest(*args, **kwargs):
        # Test the case when the dataset exists
        dataset_name_result = LabMock.mocked_request_get_dataset_name_exists(
            *args, **kwargs)
        if dataset_name_result.status_code == 404:
            return dataset_name_result

        # Test the case when the snapshot exists
        snapshot_name_result = LabMock.mocked_request_get_dataset_snapshot_exists(
            *args, **kwargs)
        if snapshot_name_result.status_code == 404:
            return snapshot_name_result
        list_url_parsed = LabMock.filter_url(args[0])
        dataset_name = list_url_parsed[2]
        snapshot_name = list_url_parsed[4]
        # return a minimal snapshot manifest
        return DotMap({'status_code': 200,
                       'content': {"name": 'manifest',
                                   "dataset": dataset_name,
                                   "snapshot": snapshot_name}})

    @staticmethod
    def mocked_request_get_create_snapshot(*args, **kwargs):
        dataset_name_result = LabMock.mocked_request_get_dataset_name_exists(
            *args, **kwargs)
        if dataset_name_result.status_code == 404:
            return dataset_name_result

        return DotMap({'text': LabMock.html_with_csrf})

    @staticmethod
    def mocked_request_session_post(*args, **kwargs):
        if not LabMock.regex_config['snapshot_pattern'].match(args[0]):
            return DotMap({'status_code': 404})

        # Test the case when the dataset does not exists
        dataset_name_result = LabMock.mocked_request_get_dataset_name_exists(
            *args, **kwargs)
        if dataset_name_result.status_code == 404:
            return dataset_name_result

        # Check the post request parameters
        data = kwargs.get('data', 0)
        if data == 0:
            return Exception("The post request does not contain parameters")
        if not data['csrf-token'] == LabMock.config['csrf_token']:
            return Exception("crsf token mismatch")

        # modify the url to include the name of the snapshot
        new_url = args[0] + '/' + data['dataset_snapshot[tag_query]']
        args = list(args)
        args[0] = new_url
        args = tuple(args)

        # Test the case when the dataset exists and the snapshot does not
        snapshot_name_result = LabMock.mocked_request_get_dataset_snapshot_exists(
            *args, **kwargs)
        if snapshot_name_result.status_code == 200:
            return DotMap({'status_code': 404})

        # Create the snapshot in the config
        list_url_parsed = LabMock.filter_url(args[0])
        dataset_name = list_url_parsed[2]
        snapshot_name = list_url_parsed[4]
        LabMock.config['snapshots'][dataset_name].append(snapshot_name)
        return DotMap({'status_code': 200})


class LabTests(unittest.TestCase):
    def setUp(self):
        # generate random csrf_token
        LabMock.config['csrf_token'] = ''.join(
            random.choice(string.ascii_letters) for i in range(20))
        # insert csrf token into html_template
        LabMock.html_with_csrf = LabMock.html_with_csrf_template.format(
            LabMock.config['csrf_token'])
        # reset existing dataset list
        LabMock.config['dataset'] = ['dataset_{}'.format(i) for i in range(10)]
        # reset existing snapshots for each data set
        LabMock.config['snapshots'] = dict(
            (d, ['snapshot_{}'.format(i) for i in 'abc']) for d in LabMock.config['dataset'])
        # reset existing images for each data set
        LabMock.config['images'] = dict(
            (d, ['image_{}'.format(i) for i in 'abc']) for d in LabMock.config['dataset'])
        # reset existing ontology list
        LabMock.config['ontology'] = [
            'ontology_{}'.format(i) for i in range(10)]

    def create_patched_auth_ob(self):
        """ Create a lab object with a valid authentification key
        """
        with patch('studio.settings.lab.requests.get', side_effect=LabMock.mocked_request_get):
            ob = lab.Lab(LabMock.config['api_key'])
        return ob

    @patch('studio.settings.lab.requests.get', side_effect=LabMock.mocked_request_get)
    def test_auth(self, mock_get):
        # Test the case where the lab api key is valid
        ob = lab.Lab(LabMock.config['api_key'])
        self.assertTrue(ob.auth())

        # Test the case where the lab api key is wrong
        try:
            lab.Lab("Wrong_api_key")
        except Exception as e:
            if str(e).startswith("Your lab api key"):
                self.assertFalse(False)

    @patch('studio.settings.lab.requests.get', side_effect=LabMock.mocked_request_get)
    def test_dataset_name_exists(self, mock_get_dataset):
        try:
            ob = self.create_patched_auth_ob()
        except Exception as e:
            if str(e).startswith("Your lab api key"):
                self.assertFalse(True, "Wrong api token in non auth test")

        # Test the case where the dataset exists
        dataset_exists = ob.dataset_name_exists('dataset_1')
        self.assertTrue(dataset_exists)

        # Test the case where the dataset does not exist
        dataset_exists = ob.dataset_name_exists('dataset_does_not_exist')
        self.assertFalse(dataset_exists)

    @patch('studio.settings.lab.requests.get', side_effect=LabMock.mocked_request_get)
    def test_dataset_snapshot_exists(self, mock_get_snapshot):
        try:
            ob = self.create_patched_auth_ob()
        except Exception as e:
            if str(e).startswith("Your lab api key"):
                self.assertFalse(True, "Wrong api token in non auth test")
        # Test the case where the snapshot exists
        snapshot_set_exists = ob.dataset_snapshot_exists(
            'snapshot_a', 'dataset_1')
        self.assertTrue(snapshot_set_exists)

        # Test the case where the dataset does not exist
        snapshot_set_exists = ob.dataset_snapshot_exists(
            'snapshot_5', 'dataset_t')
        self.assertFalse(snapshot_set_exists)

        # Test the case where the snapshot does not exist
        snapshot_set_exists = ob.dataset_snapshot_exists(
            'snapshot_5', 'dataset_1')
        self.assertFalse(snapshot_set_exists)

    @patch('studio.settings.lab.requests.get', side_effect=LabMock.mocked_request_get)
    def test_dataset_image_exists(self, mock_get_image):
        try:
            ob = self.create_patched_auth_ob()
        except Exception as e:
            if str(e).startswith("Your lab api key"):
                self.assertFalse(True, "Wrong api token in non auth test")

        # Test the case where the image exists
        image_set_exists = ob.image_exists('dataset_1', 'image_a')

        self.assertTrue(image_set_exists)

        # Test the case where the image does not exist
        image_set_exists = ob.image_exists('dataset_1', 'image_x')
        self.assertFalse(image_set_exists)

    @patch('studio.settings.lab.requests.get', side_effect=LabMock.mocked_request_get)
    def test_ontology_exists(self, mock_get_ontology):
        try:
            ob = self.create_patched_auth_ob()
        except Exception as e:
            if str(e).startswith("Your lab api key"):
                self.assertFalse(True, "Wrong api token in non auth test")
        # Test the case where the ontology exist
        ontology_exists = ob.ontology_exists('ontology_1')
        self.assertTrue(ontology_exists)

        # Test the case where the ontology does not exist
        ontology_exists = ob.ontology_exists('ontology_does_not_exist')
        self.assertFalse(ontology_exists)

    @patch('builtins.open.write')
    @patch('builtins.open')
    @patch('studio.settings.lab.requests.get', side_effect=LabMock.mocked_request_get)
    def test_download_snapshot_manifest(self, mock_get_manifest, mock_open, mock_open_write):
        try:
            ob = self.create_patched_auth_ob()
        except Exception as e:
            if str(e).startswith("Your lab api key"):
                self.assertFalse(True, "Wrong api token in non auth test")

        # Test the case where dataset does not exist
        snapshot_set_exists = ob.download_snapshot_manifest(
            'snapshot_5', 'dataset_t', 'test')
        self.assertFalse(snapshot_set_exists)

        # Test the case where the snapshot exists
        snapshot_set_exists = ob.download_snapshot_manifest(
            'snapshot_a', 'dataset_1', 'test')
        self.assertTrue(snapshot_set_exists)

        # Test the case where the snapshot does not exist
        snapshot_set_exists = ob.download_snapshot_manifest(
            'snapshot_5', 'dataset_1', 'test')
        self.assertFalse(snapshot_set_exists)

    @patch('studio.settings.lab.requests.Session.post', side_effect=LabMock.mocked_request_session_post)
    @patch('studio.settings.lab.requests.get', side_effect=LabMock.mocked_request_get)
    def test_create_dataset_snapshot(self, mock_get, mock_post):
        try:
            ob = self.create_patched_auth_ob()
        except Exception as e:
            if str(e).startswith("Your lab api key"):
                self.assertFalse(True, "Wrong api token in non auth test")

        # Test the case where the dataset exists but snapshot does not => add snapshot
        dataset_name = "dataset_1"
        snapshot_name = "snapshot_k"

        ob.create_dataset_snapshot(
            dataset_name, snapshot_name, snapshot_name)

        exist = snapshot_name in LabMock.config['snapshots'][dataset_name]
        self.assertTrue(exist)

        # Test the case where both the dataset and the snapshot exist
        dataset_name = "dataset_1"
        snapshot_name = "snapshot_k"
        try:
            ob.create_dataset_snapshot(
                dataset_name, snapshot_name, snapshot_name)
        except Exception as e:
            if str(e).startswith("snapshot_already_exists"):
                self.assertFalse(False)

        # Test the case where the dataset does not exist
        dataset_name = "dataset_does_not_exist"
        snapshot_name = "snapshot_k"
        try:
            ob.create_dataset_snapshot(
                dataset_name, snapshot_name, snapshot_name)
        except Exception as e:
            if str(e) == "The dataset `{}` doesn't exist".format(dataset_name):
                self.assertFalse(False)
