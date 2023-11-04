# Fashion-flow
Fashion-flow allows users to visualize how clothing items will look on them without physically trying them on. This project aims to perform virtual try-on using the ViTOn HD Dataset.
## Dataset
The ViTOn HD Dataset is used in this project. Please refer to the dataset's official website for details on access and usage.
https://github.com/shadow2496/VITON-HD

Along with the base dataset we have done some custom preprocessing to improve the results of the model. The preprocessed versions of the data along with the pretrained weights of the models for converting an image to agnostic and generator and discriminator have been uploaded in the below drive link.

https://drive.google.com/drive/u/2/folders/1He7jVGmASwuPSPkkXOdURrgMyGFmSqyl
## Preprocessing
The `preprocess` folder contains the code for skeleton pose and agnostic preprocessing. The respective files conatins the function which takes the image as input and returns the preprocessed image as output. The code for face segment is adapted from a different GitHub repository. You can set it up following the below steps.
### Dependencies
* [git-lfs](https://git-lfs.github.com/)
* [Numpy](https://www.numpy.org/): `$pip3 install numpy`
* [OpenCV](https://opencv.org/): `$pip3 install opencv-python`
* [PyTorch](https://pytorch.org/): `$pip3 install torch torchvision`
* [ibug.roi_tanh_warping](https://github.com/ibug-group/roi_tanh_warping): See this repository for details: [https://github.com/ibug-group/roi_tanh_warping](https://github.com/ibug-group/roi_tanh_warping).
* [ibug.face_detection](https://github.com/hhj1897/face_detection) (only needed by the test script): See this repository for details: [https://github.com/hhj1897/face_detection](https://github.com/hhj1897/face_detection).

### Steps

```bash
git clone https://github.com/hhj1897/face_parsing
cd face_parsing
git lfs pull
pip install -e .

-> replace face_parsing_test.py with the file attached in the repository

'''bash
python face_warping_test.py -i 0 -e rtnet50 --decoder fcn -n 11 -d cuda:0
## Training
Hello ji
## Testing
