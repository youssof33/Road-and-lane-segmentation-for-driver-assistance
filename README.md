17. Road and lane segmentation for driver assistance
Mark the drivable road and the lane markings in a dashcam video, which is the first stage of a lane departure
warning.
Course path. A colour based baseline, then FCN, then U-Net, then a test on night and rain footage where the
baseline collapses and the network degrades more gracefully.
Data and demo. KITTI Road or Cityscapes for training, plus video recorded locally. The demo is a processed
clip with a lane departure alert.