import numpy as np
import scipy.io
import sys
import argparse

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, Dropout, Reshape
from keras.layers.recurrent import LSTM
from keras.utils import np_utils, generic_utils
from keras.callbacks import ModelCheckpoint, RemoteMonitor

from sklearn.externals import joblib
from sklearn import preprocessing

from spacy.en import English

from utils import grouper, selectFrequentAnswers
from features import get_images_matrix, get_answers_matrix, get_questions_tensor_timeseries


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-num_hidden_units_mlp', type=int, default=1024)
	parser.add_argument('-num_hidden_units_lstm', type=int, default=512)
	parser.add_argument('-num_hidden_layers_mlp', type=int, default=3)
	parser.add_argument('-dropout', type=float, default=0.5)
	parser.add_argument('-activation_mlp', type=str, default='tanh')
	#TODO Feature parser.add_argument('-language_only', type=bool, default= False)
	args = parser.parse_args()

	word_vec_dim= 300
	img_dim = 4096
	max_len = 30
	nb_classes = 1000

	#get the data
	questions_train = open('../data/preprocessed/questions_train2014.txt', 'r').read().decode('utf8').splitlines()
	questions_lengths_train = open('../data/preprocessed/questions_lengths_train2014.txt', 'r').read().decode('utf8').splitlines()
	answers_train = open('../data/preprocessed/answers_train2014.txt', 'r').read().decode('utf8').splitlines()
	images_train = open('../data/preprocessed/images_train2014.txt', 'r').read().decode('utf8').splitlines()
	vgg_model_path = '../features/coco/vgg_feats.mat'

	maxAnswers = 1000
	questions_train, answers_train, images_train = selectFrequentAnswers(questions_train,answers_train,images_train, maxAnswers)
	questions_lengths_train, questions_train, answers_train, images_train = (list(t) for t in zip(*sorted(zip(questions_lengths_train, questions_train, answers_train, images_train))))

	#encode the remaining answers
	labelencoder = preprocessing.LabelEncoder()
	labelencoder.fit(answers_train)
	nb_classes = len(list(labelencoder.classes_))
	joblib.dump(labelencoder,'../models/labelencoder.pkl')
	#defining our LSTM based model
	image_model = Sequential()
	image_model.add(Reshape(input_shape = (img_dim,), dims=(img_dim,)))
	#print image_model.output_shape
	#512 hidden units in LSTM layer. 300-dimnensional word vectors.
	language_model = Sequential()
	language_model.add(LSTM(output_dim = args.num_hidden_units_lstm, return_sequences=False, input_shape=(max_len, word_vec_dim)))
	#print language_model.output_shape

	model = Sequential()
	model.add(Merge([language_model, image_model], mode='concat', concat_axis=1))
	print model.output_shape
	for i in xrange(args.num_hidden_layers_mlp):
		model.add(Dense(args.num_hidden_units_mlp, init='uniform'))
		model.add(Activation(args.activation_mlp))
		model.add(Dropout(args.dropout))

	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	json_string = model.to_json()
	model_file_name = '../models/lstm_1_num_hidden_units_lstm_' + str(args.num_hidden_units_lstm) + '_num_hidden_units_mlp_' + str(args.num_hidden_units_mlp) + '_num_hidden_layers_mlp_' + str(args.num_hidden_layers_mlp)
	open(model_file_name + '.json', 'w').write(json_string)

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	print 'Compilation done'

	features_struct = scipy.io.loadmat(vgg_model_path)
	VGGfeatures = features_struct['feats']
	print 'loaded vgg features'
	image_ids = open('../features/coco_vgg_IDMap.txt').read().splitlines()
	img_map = {}
	for ids in image_ids:
		id_split = ids.split()
		img_map[id_split[0]] = int(id_split[1])

	nlp = English()
	print 'loaded word2vec features...'
	## training
	print 'Training started...'
	numEpochs = 100
	model_save_interval = 10
	batchSize = 128
	for k in xrange(numEpochs):

		progbar = generic_utils.Progbar(len(questions_train))

		for qu_batch,an_batch,im_batch in zip(grouper(questions_train, batchSize, fillvalue=questions_train[0]), 
												grouper(answers_train, batchSize, fillvalue=answers_train[0]), 
												grouper(images_train, batchSize, fillvalue=images_train[0])):
			timesteps = len(nlp(qu_batch[-1])) #questions sorted in descending order of length
			X_q_batch = get_questions_tensor_timeseries(qu_batch, nlp, timesteps)
			X_i_batch = get_images_matrix(im_batch, img_map, VGGfeatures)
			Y_batch = get_answers_matrix(an_batch, labelencoder)
			loss = model.train_on_batch([X_q_batch, X_i_batch], Y_batch)
			progbar.add(batchSize, values=[("train loss", loss)])

		
		if k%model_save_interval == 0:
			model.save_weights(model_file_name + '_epoch_{:03d}.hdf5'.format(k))

	model.save_weights(model_file_name + '_epoch_{:03d}.hdf5'.format(k))
	
if __name__ == "__main__":
	main()