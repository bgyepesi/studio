import os
import GPUtil
import warnings


class DGX(object):
    # This class will be responsible of any type of interaction with the DGX infrastructure,
    # like allocating the GPU resources or handling the creation of symbolic datasets.

    def __init__(self):
        self.num_gpus = None

    @staticmethod
    def available_GPUs(max_GPUs):
        """
        Return a list with available GPU devices.
        Args:
            max_GPUs: Maximum number of GPUs to be assigned if available.
        Return:
            availableIDs: List of available GPU devices.
        """
        availableIDs = GPUtil.getAvailable(order='first', limit=max_GPUs, maxLoad=0.5, maxMemory=0.5, includeNan=False,
                                           excludeID=[], excludeUUID=[])
        return availableIDs

    def allocate_GPUs(self, num_gpus, max_GPUs):
        """
        Allocate GPU devices based on the requested 'num_gpus' and their current availability.

        Args:
            num_gpus: integer indicating the number of requested GPUs.
            max_GPUs: integer indicating the maximum number of GPUs to be assigned if available.
        Return:
            allocated_GPUs: list of allocated devices.
        """
        # Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi.
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        self.num_gpus = num_gpus

        if self.num_gpus:
            # Get the available GPUs
            availableIDs = self.available_GPUs(max_GPUs)

            if not availableIDs:
                # TODO: If all devices are busy, submit job to pool. For now, just raise an error.
                raise ValueError('All the GPU devices are being used.')
            elif len(availableIDs) < self.num_gpus:
                warnings.warn('Fewer GPU devices than requested are available.', UserWarning)
                # Set CUDA_VISIBLE_DEVICES to available deviceIDs
                allocated_GPUs = ','.join(str(x) for x in availableIDs)
                os.environ["CUDA_VISIBLE_DEVICES"] = allocated_GPUs
            else:
                deviceIDs = availableIDs[:self.num_gpus]  # grab first num_gpus elements from availableIDs
                # Set CUDA_VISIBLE_DEVICES to num_gpus availableIDs
                allocated_GPUs = ','.join(str(x) for x in deviceIDs)
                os.environ["CUDA_VISIBLE_DEVICES"] = allocated_GPUs

            print('Running code on GPU(s): ', allocated_GPUs)
            return list(allocated_GPUs)
        else:
            print('GPUs have not been requested. Running code on CPU.')
            return None

    # def create_symbolic_dataset(self, dataset_manifest):
    #     # dataset_manifest: dictionary containing {'original_image_id', 'symbolic_folder_name'}
    #     return symbolic_path
