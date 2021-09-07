import os

from gcloud import storage


class GCloud(object):
    # This class will be responsible of handling GCloud services
    def __init__(self):
        self.credentials = None
        self.project = None


class GStorage(GCloud):
    # This GCloud subclass is responsible of handling GStorage services
    # To authorize user, please run `gcloud auth application-default login`
    def __init__(self, project, bucket):
        self.project = project
        try:
            self.client = storage.Client(project=self.project)
        except Exception:
            raise ValueError("GCloud authentification failed. Please run `gcloud auth application-default login` in your terminal.")

        self.bucket = self.client.get_bucket(bucket)

    def upload_file(self, origin_file, dest_file):
        """
        Uploads `origin_file` to the bucket as `dest_file`.

        Args:
            origin_folder: (string) local path to the origin file
            dest_folder: (string) bucket path to the destination file
        """
        blob = self.bucket.blob(dest_file)
        print("Uploading " + str(origin_file) + '...')
        blob.upload_from_filename(origin_file)

    def upload_folder(self, origin_folder, dest_folder):
        """
        Uploads `origin_folder` recursively to the bucket as `dest_folder`.

        Args:
            origin_folder: (string) local path to the origin folder
            dest_folder: (string) bucket path to the destination folder
        """
        for root, dirs, files in os.walk(origin_folder):
            for file in files:
                origin_file = os.path.join(root, file)
                relative_path = os.path.relpath(origin_file, os.path.dirname(origin_folder))
                dest_file = os.path.join(dest_folder, relative_path)
                self.upload_file(origin_file, dest_file)
