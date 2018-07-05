# asciiko
--------------------------------------------------------------------------------
A deep ascii art generator. This is an on-going project, it's not finished.

## Requirements
* Python 2/3
* Install Pytorch, follow [this][pytorch_install].
* Install OpenCV2, `conda install opencv` should work.
* Ask for permission of using ETL datasets from [here][etlcdb], then unzip and put ETL1 or ETL6 into char_classifier folder (we recommend ETL6, which contains more punctuation marks, which might be helpful for generating ascii art).

## First Time Running
* It preprocesses and parses the ETL datasets, save the useful part into .npy files.
* It will take some time when first time running.

## To Run
* a pretrained model is provided in `saved_models/`.
* In `char_classifier/config/config.yaml`, enable CUDA if you have access to it, also specify which version of ETL are you using.
* To train a model, run `python char_classifier/train.py -c char_classifier/config/`.
* To generate ascii strings from an image, run `python char_classifier/img2charid.py`, it will generate a `.json` file;
* To generate ascii strings from a video, run `python char_classifier/imgs2charids.py`, it will generate a `.json` file.
* To render the ascii iamge, run `python char_classifier/renderer.py`, with your `.json` file specified inside.

<p align=center><img width="80%" src="demo.jpg" /></p>

## Authors
Eric Yuan, Tatsuro Oya, Saku Sugawara

## LICENSE
[GLWTPL][goodluck]

[pytorch_install]: http://pytorch.org/
[etlcdb]: http://etlcdb.db.aist.go.jp/
[goodluck]: https://github.com/xingdi-eric-yuan/asciiko/blob/master/LICENSE