# Update

_Update 2._ __Check out SimpleTransformers if you want a ready to use (3 lines total for initializing, training, and evaluating) library. Currently supports BERT, RoBERTa, XLM, XLNet, and DistilBERT models for binary and multiclass classification.__

_Update 1._ __In light of the update to the library used in this repo (HuggingFace updated the `pytorch-pretrained-bert` library to [pytorch-transformers](https://github.com/huggingface/pytorch-transformers)), I have written a new [guide](https://medium.com/@chaturangarajapakshe/https-medium-com-chaturangarajapakshe-text-classification-with-transformer-models-d370944b50ca) as well as a new [repo](https://github.com/ThilinaRajapakse/pytorch-transformers-classification). I highly recommend using those instead as the code has been cleaned up both on my end and in the Pytorch-Transformers library, greatly streamlining the whole process. Also, any updates and fixes will primarily be for the updated versions. The new repo also supports XLNet, XLM, and RoBERTa models out of the box, in addition to BERT, as of September 2019.__

# Binary Text Classification with BERT
Accompanying code for the Medium article found at https://medium.com/@chaturangarajapakshe/a-simple-guide-on-using-bert-for-text-classification-bbf041ac8d04.

## Important!

This repository was not meant to be a standalone guide to using BERT. It was originally created to accompany the article above.  
If you are using this repo to set up BERT and you run into any issues, I highly recommend that you go through the article as it explains things in more detail.  
Even if you don't run into problems, the article should give a better picture of how I did things, and why I chose to do it that way. Gaining that perspective may help you to adapt this code to your project in a more suitable and/or efficient manner.

Happy coding!
