''' Imports '''
import glob
import numpy
import pickle
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from music21 import converter, instrument, note, chord  # python music

MUSIC_PATH = './music_data/*.mid'  # path to midi files directory
NOTES_PATH = './notes'  # path to file containing all notes
SAVE_PATH = './my_model'


def get_notes():
  notes = []
  for file in glob.glob(MUSIC_PATH):
    print(file)
    cur_song = converter.parse(file)

    print(f'Current song: {file}')

    cur_notes = None

    try:  # instrumental notes
      instr = instrument.partitionByInstrument(cur_song)
      cur_notes = instr.parts[0].recurse()
    except:  # flat notes
      cur_notes = cur_song.flat.notes

    for item in cur_notes:
      if isinstance(item, note.Note):  # single note to append
        notes.append(str(item.pitch))
      elif isinstance(item, chord.Chord):  # chords have multiple notes
        notes.append('.'.join(str(i) for i in item.normalOrder))
    
  with open(NOTES_PATH, 'wb') as path:
      pickle.dump(notes, path)
  return notes


def prep_data(notes, num_pitches):
  ''' convert our string notes into integers readable by the network '''

  #  alter this to alter results
  lookback = 100

  pitches = sorted(set(n for n in notes))

  #  the mapping of pitch to integer
  note_dic = dict((note, n) for n, note in enumerate(pitches))

  training_samples = []
  training_labels = []

  for i in range(0, len(notes) - lookback, 1):
    cur_samples = notes[i:i + lookback]
    cur_label = notes[i + lookback]
    training_samples.append([note_dic[samp] for samp in cur_samples])
    training_labels.append(note_dic[cur_label])

  n_samples = len(training_samples)

  #  reshape
  training_samples = numpy.reshape(training_samples, (n_samples, lookback, 1))

  #  normalize
  training_samples = training_samples / num_pitches

  training_labels = to_categorical(training_labels)

  return training_samples, training_labels


def generate_model(train_samples, num_pitches):
  model = Sequential()
  model.add(LSTM(
      512,
      input_shape=(train_samples.shape[1], train_samples.shape[2]),
      recurrent_dropout=.3,
      return_sequences=True))
  model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))
  model.add(LSTM(512))
  model.add(BatchNorm())
  model.add(Dropout(0.3))
  model.add(Dense(256))
  model.add(Activation('relu'))
  model.add(BatchNorm())
  model.add(Dropout(0.3))
  model.add(Dense(num_pitches))
  model.add(Activation('softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

  return model


def train_model():
  notes = get_notes()

  num_pitches = len(set(notes))

  train_samples, train_labels = prep_data(notes, num_pitches)

  model = generate_model(train_samples, num_pitches)

  model.fit(train_samples, train_labels, epochs=200, batch_size=128, validation_split=.2)

  return model


if __name__ == '__main__':
  model = train_model()
  model.save(SAVE_PATH)
