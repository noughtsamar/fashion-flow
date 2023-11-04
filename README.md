# Fashion-flow
Fashion-flow allows users to visualize how clothing items will look on them without physically trying them on. This project aims to perform virtual try-on using the ViTOn HD Dataset.
## Dataset
The ViTOn HD Dataset is used in this project. Please refer to the dataset's official website for details on access and usage.
https://github.com/shadow2496/VITON-HD

Along with the base dataset we have done some custom preprocessing to improve the results of the model. The preprocessed versions of the data along with the pretrained weights of the models for converting an image to agnostic and generator and discriminator have been uploaded in the below drive link.

https://drive.google.com/drive/u/2/folders/1He7jVGmASwuPSPkkXOdURrgMyGFmSqyl
## Preprocessing
The <mark>preprocess</mark> folder contains the code for skeleton pose and agnostic preprocessing. The respective files conatins the function which takes the image as input and returns the preprocessed image as output. The code for face segment is adapted from a different GitHub repository (link mentioned in References).
## Training
## Testing
