# meta_learning_robustness

original/ folder is just backup of files, don't need to look

Files are in py/

## Setting up Data
Omniglot is included in repo
### Quickdraw
Install gsutil using the insturctions on their website

Make quickdraw diectory `mkdir data/quickdraw`

Fetch quickdraw `gsutil -m cp gs://quickdraw_dataset/full/simplified/*.npy  ./data/quickdraw`

Make mini quickdraw directory `mkdir data/mini_quickdraw`

To generate mini Quickdraw, run `python3 py/generate_miniqd.py -ns 2000 -ow`

### Flowers
Make flowers directory

Download 102 category dataset, image labels, and data splits

Unzip file


### Fungi
Make fungi directory

Download train and validation images

Unzip file