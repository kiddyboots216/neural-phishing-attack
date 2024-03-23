Dependencies:

- torch
- transformers
- pandas

First download the Enron emails dataset from https://www.kaggle.com/datasets/wcukierski/enron-email-dataset by following the instructions at https://www.kaggle.com/docs/api and unzip the file and store the resulting file as 'emails.csv' in this directory.

Next download the GPTNeoX and Tokenizer checkpoints (this requires internet usage so you likely won't be able to run it on a compute-only node) somewhere they can be accessed.

Finally, run the code.

For example usage, you can run ```python main.py --help```