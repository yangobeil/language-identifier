# Language-identifier

In this project I use various neural network architectures to predict the language that a word is written in. I scrape the web using the Wikipedia package to find words in French, English and Spanish. I then use this data to train a fully connected neural network, a recurrent neural network and two types of convolutional neural networks, with 1D convolutions. I then compare the results to see which is best and use it to make predictions.

# Dependencies

A few python packages are needed for this project. Here is the full list.
- numpy
- keras
- Wikipedia
- matplotlib

To just run the prediction, only keras is necessary.

# Files rundown

Here is the list of files included and what they contain:
- generate_data: collection of functions to collect the words from wikipedia and convert them to arrays to use in NNs
- gather_data: define words to search and create the data using the functions in generate_data
- training: copy of the notebook used on Google Colab to train the networks
- analysis: using the trained networks to see which one is the best and compare their results
- predict: make predictions using the models by taking input from the command line

I also include most of the content that I used:
- model_xxx: saved keras models (best of all epochs for each model)
- history_xxx: training history for each model
- accuracy: plot of the test accuracies for the models

The saved data is not included because the files are too big.

# To predict

If someone wants only to use the files to predict words, simply type the following line in the command line:

python predict.py word -m model

where word is the word that you want to classify and model is either FF, RNN, CNN or CNNsep.

# To train new model

To train a new model from scratch, start by changing the dictionary of words to search in the file gather_data. Then train the run the rest almost as is. There are a few places where the number of languages and the maximum number of characters are hard coded so it is necessary to look out for that.

# Results

The main result is that I am able to achieve an accuracy of 86.5% on the test data using the CNN. More details about the model and the results can be found in my blog post: https://medium.com/@yan.gobeil/comparing-neural-network-architectures-through-language-classifier-using-google-colab-63167c18b919.
