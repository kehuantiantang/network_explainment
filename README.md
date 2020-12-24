# Network Explainment
---

This project include the network explainment method below:

- [LRP](https://arxiv.org/abs/1903.07317)
- [LIME](https://arxiv.org/abs/1602.04938)
- [CAM](http://arxiv.org/pdf/1512.04150.pdf)


In order to use this project, please follow the instruction:

- Install the library according to the requirement.txt file

```sh
pip install -r requirement.txt
```

- Prepare the weight, we put the weight in `./vector*` directory, you can also train the model by yourself:

```python
# the * respect the method in [LRP, grad_cam, LIME]
# 'image/text' is the context you want to test
python *_image/text.py 
```
The result is a little different, the `*_text.py` will generate the `*.html` file, you can open it by website, which highlight the context according to the weight. `*_image.py` will generate the `*.jpg`.

More detail parameter description please refer the `argparse` in `__main__`.


#### Train the model
##### pascal voc classification
- the script is `classification_image.py`, you can directly train by runing the commond follow:
```sh
python classification_image.py
```
The script will automatically download the Pascal VOC2009 image and train the network.

##### Text classification
- NBOW
	- Download the [ACLImdb](https://ai.stanford.edu/ amaas/data/sentiment/aclImdb v1.tar.gz) dataset into `./dataset/aclImdb` directory, and split into `train`, `test`.
	- Download the [Glove](http://nlp.stanford.edu/data/glove.840B.300d.zip) model, put it into './dataset/', `unzip *.zip`
	- Run the code in `./utils/text_utils.py` to generate the `train.vocab.txt`, `train_vec.pkl`, `test_vec.pkl` file, which means the vocabulary, word2vector train, word2vector test file.
	- Modify the root, train, test path in `classification_text.py`, also you need to modify `build_dataset` function path if you word2vector file store in different path.
	- Set proper parameter to run the `python classification_text.py`

- 1d_conv BOW
	- Prepare the dataset as NBOW
	- Uncomment the `IMDB_Raw` dataloader in `build_dataset` function
	- Uncomment the 'NBOW' model in `build_model` function
	- run use `python classification.py`

 



