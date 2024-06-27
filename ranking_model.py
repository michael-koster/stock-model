import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Assuming you have a dataset with numerical ratings and corresponding text reviews
ratings = np.array([4, 5, 2, 3, 1])  # Example numerical ratings
reviews = np.array([
    "Great product, highly recommended!",
    "Awesome experience with this product.",
    "Average quality, not satisfied.",
    "Decent product, could be better.",
    "Terrible product, don't waste your money."
])  # Example text reviews
sentiments = np.array([1, 1, 0, 0, 0])  # Example sentiment labels (1 for positive, -1 for negative, 0 for neutral)

# Split the dataset into train and test sets
reviews_train, reviews_test, ratings_train, ratings_test, sentiments_train, sentiments_test = train_test_split(
    reviews, ratings, sentiments, test_size=0.2, random_state=42
)

# Text preprocessing
max_words = 1000  # Maximum number of words to consider
max_sequence_length = 100  # Maximum length of each review
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(reviews_train)
sequences_train = tokenizer.texts_to_sequences(reviews_train)
sequences_test = tokenizer.texts_to_sequences(reviews_test)
word_index = tokenizer.word_index

# Pad sequences to have the same length
X_train = pad_sequences(sequences_train, maxlen=max_sequence_length)
X_test = pad_sequences(sequences_test, maxlen=max_sequence_length)

# Numerical feature normalization
ratings_min = ratings.min()
ratings_max = ratings.max()
ratings_train_normalized = (ratings_train - ratings_min) / (ratings_max - ratings_min)
ratings_test_normalized = (ratings_test - ratings_min) / (ratings_max - ratings_min)

# Define the neural network architecture
embedding_dim = 100  # Dimensionality of the word embeddings
lstm_units = 128  # Number of units in the LSTM layer

# Text input branch
text_input = Input(shape=(max_sequence_length,))
embedding_layer = Embedding(max_words, embedding_dim)(text_input)
lstm_layer = LSTM(lstm_units)(embedding_layer)

# Numerical input branch
numerical_input = Input(shape=(1,))
numerical_dense = Dense(32, activation='relu')(numerical_input)

# Merge the branches
merged = concatenate([lstm_layer, numerical_dense])
dense_layer = Dense(32, activation='relu')(merged)
output = Dense(3, activation='softmax')(dense_layer)  # 3 classes for sentiment prediction

# Create the model
model = Model(inputs=[text_input, numerical_input], outputs=output)

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit([X_train, ratings_train_normalized], sentiments_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate([X_test, ratings_test_normalized], sentiments_test, verbose=0)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy*100:.2f}%")