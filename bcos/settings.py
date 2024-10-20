"""
General settings. Mainly paths to data.
"""
import os

# ImageNet path
IMAGENET_PATH = os.getenv("IMAGENET_PATH")
# IMAGENET_PATH='/ptmp/mparcham/datasets/ILSVRC2012/'

CUBFull_PATH = os.getenv("CUBFull_PATH")
CUB_PATH = os.getenv("CUB_PATH")
VOC_PATH = os.getenv("VOC_PATH")
SUN397_PATH = os.getenv("SUN397_PATH")
WATERBIRDS_PATH = os.getenv("WATERBIRDS_PATH")
FLOWERS102_PATH = os.getenv("FLOWERS102_PATH")

IMN_CLASS_SUBSET_PATH = os.getenv("IMN_CLASS_SUBSET_PATH")
# ---------------------------------------------
# Following are only needed for caching!!!
# ---------------------------------------------
SHMTMPDIR = "/dev/shm"

# ImageNet train data tar files
# Note this is not the same as the official ImageNet tar file
# this expects each class to inside its own tar file, which will then be extracted
# to SHMTMPDIR
# IMAGENET_TRAIN_TAR_FILES_DIR = os.getenv(
#     "IMAGENET_TRAIN_TAR_FILES_DIR", "/ptmp/nasingh/data/ImageNet_class_tars"
# )

# Path to redis server
# REDIS_SERVER = os.getenv(
#     "REDIS_SERVER_PATH", "/raven/u/mboehle/mambaforge/envs/pl/bin/redis-server"
# )
