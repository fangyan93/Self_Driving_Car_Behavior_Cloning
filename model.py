import csv
import matplotlib.pyplot as plt
import numpy as np
import sys
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D, Dropout
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.regularizers import l2, activity_l2
from keras.utils.visualize_util import plot


# create the model
def create_model():
	model = Sequential()
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
	model.add(Cropping2D(cropping=((70,25),(0,0))))

	model.add(Conv2D(24,5,5,W_regularizer=l2(0.0001)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D())
	model.add(Dropout(0.3))

	model.add(Conv2D(36,5,5,W_regularizer=l2(0.0001)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D())
	model.add(Dropout(0.3))

	model.add(Conv2D(48,5,5))
	model.add(Activation('relu'))
	model.add(MaxPooling2D())

	model.add(Conv2D(64,3,3))
	model.add(Activation('relu'))

	model.add(Flatten())
	model.add(Dense(100,W_regularizer=l2(0.01)))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	plot(model, to_file='model.png', show_shapes=True)
	return model

# e.g dir = "./train_data/"
def generator(lines, batch_size, dire):
	length = len(lines[0])
	while 1:
		shuffle(lines)
		for offset in range(0, length, batch_size):
			batch = lines[offset:offset+batch_size]
			image = []
			measurements = []
			for line in batch:
				center_name = line[0].split('/')[-1]
				left_name = line[1].split('/')[-1]
				right_name = line[2].split('/')[-1]

				image_path = dire + "IMG/"
				center_path = image_path + center_name 
				left_path = image_path + left_name 
				right_path = image_path + right_name 

				image.append(plt.imread(center_path))
				image.append(plt.imread(left_path))
				image.append(plt.imread(right_path))

				correction = 0.2
				measurement = float(line[3])
				measurement_left = measurement + correction
				measurement_right = measurement - correction

				measurements.append(measurement)
				measurements.append(measurement_left)
				measurements.append(measurement_right)

			# print(len(image), len(measurements))
			X_train = np.array(image)
			y_train = np.array(measurements)
			yield shuffle(X_train, y_train)
	
# usage: python model.py ./directory/ model_name_for_load_weight model_name_to_save
def main():
	arg = sys.argv
	print(arg)
	directory = arg[1]
	output_name = "model.h5"
	save_file_name = output_name
	if len(arg) > 3:
		output_name = arg[2]
		save_file_name = arg[3]
	lines = []
	csv_file = directory + "driving_log.csv"
	print(csv_file)
	with open(csv_file) as log:
	    reader = csv.reader(log)
	    for line in reader:
	        lines.append(line)

	train_samples, validation_samples =  train_test_split(lines, test_size=0.3) 

	# a batch of 32 of csv lines in each generation
	train_generator = generator(train_samples, 32, directory)
	validation_generator = generator(validation_samples, 32, directory)

	model = create_model()
	try:
		model.load_weights(output_name)
	except:
		model = create_model()
	model.compile(loss='mse', optimizer='adam')
	# since we use all 3 camera images, the sample_per_epoch is 32 * 3
	model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*3, validation_data=validation_generator, nb_val_samples=len(validation_samples)*3, nb_epoch=8)

	model.save(save_file_name)

if __name__ == '__main__':
	main()
