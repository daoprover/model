import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dropout, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class LLM:
    """
    The LLM class represents a Language Model that uses LSTM (Long Short-Term Memory) neural networks for text generation. It provides methods for preprocessing data, training the model
    *, predicting the next word, and saving the model.

    Attributes:
        max_sequence_length (int): Maximum sequence length for input sequences.
        total_words (int): Total number of unique words in the dataset.
        _input_sequences (numpy.ndarray): Padded input sequences for training.

    Methods:
        __init__(self)
            Initializes the LLM class by creating the model architecture.

        preprocess(self, data)
            Preprocesses the input data by tokenizing and generating input sequences for training.

        train(self)
            Compiles and trains the model using the preprocessed input sequences.

        predict(self, data)
            Predicts the next word given an input string.

        save(self)
            Saves the trained model to a file.
    """

    def __init__(self):
        self.max_sequence_length = None
        self.total_words = 100
        self._input_sequences = None

        self._model = Sequential()
        self._model.add(Embedding(self.total_words, 100, input_length=self.max_sequence_length - 1))
        self._model.add(LSTM(150, return_sequences=True))
        self._model.add(Dropout(0.2))
        self._model.add(LSTM(100))
        self._model.add(Dense(self.total_words, activation='softmax'))

    def preprocess(self, data):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(data)
        self.total_words = len(tokenizer.word_index) + 1
        input_sequences = []
        for line in data:
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i + 1]
                input_sequences.append(n_gram_sequence)
        max_sequence_length = max([len(x) for x in input_sequences])
        self._input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

    def train(self):
        self._model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        X, y = self._input_sequences[:, :-1], self._input_sequences[:, -1]
        y = tf.keras.utils.to_categorical(y, num_classes=self.total_words)
        self._model.fit(X, y, epochs=200, verbose=1)

    def predict(self, data):
        token_list = tokenizer.texts_to_sequences([data])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding='pre')
        predicted_probabilities = self._model.predict(token_list, verbose=0)[0]
        predicted_index = np.argmax(predicted_probabilities)
        output_word = tokenizer.index_word[predicted_index]

        return output_word


    def save(self):
        self._model.save('custom_llm_model.h5')