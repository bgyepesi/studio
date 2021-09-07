#!/usr/bin/env bash

tf_model_dir=`mktemp -u`
CUDA_VISIBLE_DEVICES=''

# Set model paths
if [ "$#" -eq 2 ]; then
    # Convert to Tensorflow
    CUDA_VISIBLE_DEVICES='' keras_to_tensorflow "$1" "$tf_model_dir"
elif [ "$#" -eq 3 ]; then
    CUDA_VISIBLE_DEVICES='' keras_to_tensorflow "$1" "$tf_model_dir" "$3"
else
  echo "Usage: keras_to_tf_zip <hdf5 file> <output file> (<feature layer>)"
  exit 1
fi

# Zip files
cd "$tf_model_dir"
zip -r "$2" .

# Cleanup
rm -rf "$tf_model_dir"
