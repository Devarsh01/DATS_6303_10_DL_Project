#%%
import numpy as np
import pretty_midi
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Conv1D, BatchNormalization, LSTM, Dense, Softmax, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Check for GPU availability
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if tf.config.experimental.list_physical_devices('GPU'):
    print("Using GPU")
else:
    print("Using CPU")


def load_midi_details(directory):
    """Load all MIDI files in the directory and extract detailed note sequences."""
    all_sequences = []
    for filename in os.listdir(directory):
        if filename.endswith('.mid'):
            path = os.path.join(directory, filename)
            midi_data = pretty_midi.PrettyMIDI(path)
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    all_sequences.append({
                        'pitch': note.pitch,
                        'velocity': note.velocity,
                        'start': note.start,
                        'end': note.end
                    })
    return all_sequences

def create_input_target_sequences(sequence, seq_length):
    """Create input and target sequences from detailed note sequences."""
    input_sequences = []
    target_sequences = []
    for i in range(len(sequence) - seq_length):
        # input_seq = [[event['pitch'], event['velocity'], event['start'], event['end']] for event in sequence[i:i+seq_length]]
        # target_seq = [[event['pitch'], event['velocity'], event['start'], event['end']] for event in sequence[i+1:i+1+seq_length]]
        input_seq = [[event['pitch']] for event in sequence[i:i + seq_length]]
        target_pitch = sequence[i + seq_length]['pitch']
        target_seq = to_categorical(target_pitch, num_classes=vocab_size)
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)
    print(len(input_sequences))
    print(input_sequences[0])
    return np.array(input_sequences), np.array(target_sequences)

#%%
def build_model(seq_length, vocab_size, embedding_dim):
    inputs = Input(shape=(seq_length,))

    # Embedding layer: Converts input sequence of token indices to sequences of vectors
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=seq_length)(inputs)

    # LSTM layer: You can stack more LSTM layers or adjust the number of units
    x = LSTM(units=256, return_sequences=True)(x)  # return_sequences=True if next layer is also RNN
    x = LSTM(units=256)(x)  # Last LSTM layer does not return sequences

    # Output layer: Linear layer (Dense) with 'vocab_size' units to predict the next pitch
    outputs = Dense(vocab_size, activation='softmax')(x)  # Using softmax for output distribution

    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

seq_length = 30  # Length of the input sequences
vocab_size = 128  # Number of unique pitches (for MIDI, typically 128)
embedding_dim = 100 # Define the length of sequences for input and target

main_directory = '/home/ubuntu/Devarsh_DL/Project/DL_Dataset/'

for file in os.listdir(main_directory):
    print(file)
    directory = f'/home/ubuntu/Devarsh_DL/Project/DL_Dataset/{file}/'
    print(directory)
    sequences = load_midi_details(directory)
    input_sequences, target_sequences = create_input_target_sequences(sequences, seq_length)
    model = build_model(seq_length, vocab_size, embedding_dim)
    model.summary()
    model.fit(input_sequences, target_sequences, epochs=50, batch_size=32)
    model.save(f'/home/ubuntu/Devarsh_DL/Project/mddels/{file}.h5')
    print(f'The model for {file} has been saved!!!!!!!!!!')
    print("--------------------------------------------------------")

#%%

# Load and preprocess data


sequences = load_midi_details(directory)
input_sequences, target_sequences = create_input_target_sequences(sequences, seq_length)


print("Input sequences shape:", input_sequences.shape)
print("Target sequences shape:", target_sequences.shape)