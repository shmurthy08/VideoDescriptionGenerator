# VideoDescriptionGenerator

The goal of this project is to create a model, leveraging ConvLSTM and Transformers to produce a model that generates simple, yet effective captions (descriptions) of videos. We will utilize and evaluate the [MSVD dataset](https://paperswithcode.com/dataset/msvd) to determine if our model is an effective model that can further the research done within this domain.

# Authors:
- Shree Murthy (shmurthy@chapman.edu)
- Dylan Inafuku (dinafuku@chapman.edu)
- Rahul Sura (sura@chapman.edu)
- Shanzeh Bandukda (bandukda@chapman.edu)
- Noah Fuery (nfuery@chapman.edu)

# Final Presentation:

[Final Presentation](https://docs.google.com/presentation/d/1Fi8m1O6YEa9I6XqGIBWcjJpIPgreoqhprWltktRnhKY/edit?usp=sharing)


## Model weights:
- The ConvTransformer model weights are extremely large and unable to be uploaded to github for reproduction.
- However, the feature extraction model has been uploaded to github, the ConvTransformer model will leverage this feature extraction model to create a close 1:1 reproduction of the model we trained. Furthermore, the model was trained with the best parameters using Keras Tuner, hence the weights and gradients are inherently the best possible version. 
- We tried multiple times to upload the weights to github, however the file couldn't be read and published since it was too large. We apologize for this issue, and we hope providing the feature extraction model will suffice coupled with the best possible parameters for the ConvTransformer model.

## Project Contributons:
- NOTE: Everyone contributed to the coding aspects via pair programming. 
- Shree Murthy: Coded the Feature Extraction model and Transformer models.
- Dylan Inafuku: Wrote the report and helped debug models/methods
- Rahul Sura: Created Preprocessing steps and Data Splitting steps
- Shanzeh Bandukda: Worked with Noah to research metrics and informed Shree how to implement them into the code. Created slides for the presentation
- Noah Fuery: Worked with Shanzeh to research metrics. Helped Dylan write the report (i.e Introduction and Abstract)

## Install necessary libraries
```bash
chmod 755 install.sh
./install.sh
```

## Steps to run on Linux-based environment:
```bash
cd src/
python3 preprocessing.py
python3 data_splitting.py
python3 feature_extraction.py
python3 conv_transformer.py
```
## Location of code:
- All code is located in the `src/` directory.
- All data is located in the `data/` directory.

## References:
- [MSVD dataset](https://paperswithcode.com/dataset/msvd)
- [Image Captioning with Transformer](https://keras.io/examples/vision/image_captioning/)
- [TF Math](https://www.tensorflow.org/api_docs/python/tf/math)
- [TensorBoard](https://www.tensorflow.org/tensorboard)
- [Tensor to Numpy](https://saturncloud.io/blog/convert-a-tensor-to-a-numpy-array-in-tensorflow/)
- [Understanding AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
- [LRScheduler](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/LearningRateSchedule)
- [Model Subclass API - TF](https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing)