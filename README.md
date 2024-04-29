# VideoDescriptionGenerator

The goal of this project is to create a model, leveraging ConvLSTM and Transformers to produce a model that generates simple, yet effective captions (descriptions) of videos. We will utilize and evaluate the [MSVD dataset](https://paperswithcode.com/dataset/msvd) to determine if our model is an effective model that can further the research done within this domain.

## Project Contributors:
- Shree Murthy
- Dylan Inafuku
- Rahul Sura
- Shanzeh Bandukda
- Noah Fuery

## Install necessary libraries
```bash
chmod 755 install.sh
install.sh
```

## Steps to run on Linux-based environment:
```bash
cd src/
python3 preprocessing.py
python3 data_splitting.py
python3 model.py
```