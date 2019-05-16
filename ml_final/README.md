## The Data

The dataset I used was from Kaggle (https://www.kaggle.com/kostastokis/simpsons-faces), but I did have to manually filter through the images and select only Homer. If any are interested in _just_ the photos of Homer Simpson I can post them on Kaggle as well.

File path may need changing depending on where you place the dataset.

## To Run 

`python3 Lupini_DCGAN.py`
Some lines should be commented/uncommented if running from the terminal. In the `plot_images()` function, terminal users should comment out `plt.show()` and uncomment the lines that save images into a file. 

Or...

`jupyter notebook` while in the directory with the .ipybn file

## Credit Where Credit is Due

DCGAN formation was taken from the Keras Deep Learning Cookbook Chapter 6 by Rajdeep Dua, and Manpreet Singh Ghotra. (https://www.packtpub.com/big-data-and-business-intelligence/keras-deep-learning-cookbook)
