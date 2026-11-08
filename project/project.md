Algorithm 1: Color-based segmentation (baseline)

- Python.
- OpenCV.
- HSV color space.
- Color thresholding.
- Gaussian blur.
- Canny edge detection.
- Hough Line Transform.
How does it work?
1. Read a frame from the dashcam.
2. Convert the image from BGR to HSV.
3. Identify white and yellow lane markings using color thresholds.
4. Apply a region of interest to ignore the sky and unnecessary objects.
5. Use edge detection and Hough Transform to identify lane lines.
6. Apply color masks to highlight the road and lanes.
The limitation: It struggles with shadows, nighttime, rain, faded lane markings, and different road colors.

That's exactly why we need deep learning.
----------------------------------------------------------------------------------
Algorithm 2: FCN (Fully Convolutional Network)

This is your first deep-learning segmentation model.

Unlike the color-based method, FCN learns to recognize road pixels and lane-marking pixels from labeled training images.

- Convolutional layers.
- Feature extraction.
- Pooling.
- Upsampling.
- Pixel-wise classification.
- Loss functions.
model should produce a segmentation mask for each input image.
----------------------------------------------------------------------------------
Algorithm 3: U-Net

U-Net is another semantic segmentation architecture.

Its important feature is skip connections, which transfer spatial information from the encoder to the decoder.

This helps preserve details such as:
- Thin lane markings.
- Road boundaries.
- Small objects and edges.
For this project, U-Net will be the main model to compare against FCN.
Metric	What it measures
IoU	How accurately predicted pixels overlap with ground truth
Dice coefficient	Similarity between predicted and actual masks
Precision	How many predicted positive pixels are correct
Recall	How many actual positive pixels were detected
FPS	How many frames the system processes per second

final experiment should compare the three methods under daytime, nighttime, and rainy conditions.