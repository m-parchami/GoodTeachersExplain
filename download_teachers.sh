R34_BCOS_TEACHER=experiments/ImageNet/bcos_final/resnet_34
mkdir -p $R34_BCOS_TEACHER
wget -P $R34_BCOS_TEACHER https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/resnet_34-a63425a03e.pth

D169_BCOS_TEACHER=experiments/ImageNet/bcos_final/densenet_169
mkdir -p $D169_BCOS_TEACHER
wget -P $D169_BCOS_TEACHER https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/densenet_169-7037ee0604.pth
