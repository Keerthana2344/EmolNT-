#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Now, we have to tokenize text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)

X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq)

# Split data into train and test sets
X_train_pad, X_test_pad, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

# Define the deep learning model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=50),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

# Train the model
model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Evaluate the model
dl_loss, dl_mse = model.evaluate(X_test_pad, y_test)
dl_rmse = np.sqrt(dl_mse)
print("Deep Learning RMSE:", dl_rmse)

