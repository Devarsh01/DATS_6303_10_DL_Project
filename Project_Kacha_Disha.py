import os
import pretty_midi

# Path to the folder containing MIDI files
base_path = "/home/ubuntu/Deep-Learning/Disha_Kacha_DL/Project/archive/albeniz"

# List all MIDI files in the folder
midi_files = [os.path.join(base_path, file) for file in os.listdir(base_path) if file.endswith('.mid')]

# Initialize a PrettyMIDI object to store merged MIDI data
merged_midi_data = pretty_midi.PrettyMIDI()

# Iterate over each MIDI file and adjust its timing to start after the previous MIDI file
current_time = 0  # Track the current time position
for midi_file in midi_files:
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        # Shift the timing of notes in the current MIDI file to start after the current time
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                note.start += current_time
                note.end += current_time
            merged_midi_data.instruments.append(instrument)
        # Update the current time to the end of the current MIDI file
        current_time += midi_data.get_end_time()
    except Exception as e:
        print(f"Error processing {midi_file}: {e}")

# Save the merged MIDI data to a new MIDI file
merged_midi_file_path = '/home/ubuntu/Deep-Learning/Disha_Kacha_DL/Project/archive/concatenated_midi_albeniz.mid'
merged_midi_data.write(merged_midi_file_path)

# List all MIDI files in the folder
midi_files = [os.path.join(base_path, file) for file in os.listdir(base_path) if file.endswith('.mid')]

# Store individual durations and total duration
individual_durations = []
total_duration = 0

# Calculate individual durations
for midi_file in midi_files:
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        duration = midi_data.get_end_time()
        individual_durations.append(duration)
        total_duration += duration
    except Exception as e:
        print(f"Error processing {midi_file}: {e}")

# Print sum of individual durations
sum_individual_duration = sum(individual_durations)
print(f"Sum of individual MIDI file durations: {sum_individual_duration} seconds")

# Print total duration of concatenated MIDI files
print(f"Total duration of concatenated MIDI files: {total_duration} seconds")

import music21
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Function to process MIDI file and extract note sequences
def process_midi(midi_file, sequence_length=20):
    notes = []
    for element in midi_file.flat.notes:
        notes.append((element.pitch.ps, element.volume.velocity, element.duration.quarterLength))
    note_sequences = []
    for i in range(len(notes) - sequence_length):
        sequence = notes[i:i+sequence_length]
        note_sequences.append(sequence)
    return note_sequences

# Function to generate music using the trained model
def generate_music(model, initial_sequence, length=100):
    generated_sequence = initial_sequence.copy()

    # Generate additional notes based on the initial sequence
    for _ in range(length):
        # Predict the next note based on the current sequence
        prediction = model.predict(np.array([generated_sequence]))[0]

        # Append the predicted note to the generated sequence
        generated_sequence = np.vstack([generated_sequence, prediction])

    return generated_sequence

# Load the concatenated MIDI file
concatenated_midi_file_path = '/home/ubuntu/Deep-Learning/Disha_Kacha_DL/Project/archive/concatenated_midi_albeniz.mid'

try:
    # Attempt to load the MIDI file using music21
    midi_file = music21.converter.parse(concatenated_midi_file_path)

    # Process the MIDI file to extract note sequences
    note_sequences = process_midi(midi_file)

    # Convert note sequences to numpy arrays or further processing
    note_sequences = np.array(note_sequences)

    # Check if there are any note sequences extracted
    if len(note_sequences) == 0:
        raise ValueError("No note sequences found in the MIDI file")

    # Split the data into input sequences and target notes
    input_sequences = note_sequences[:, :-1]  # All but the last note
    target_notes = note_sequences[:, -1]     # The last note

    # Reshape input sequences to match LSTM input shape
    input_sequences = input_sequences.reshape(input_sequences.shape[0], input_sequences.shape[1], 3)

    # Build the RNN model
    model = Sequential([
        LSTM(256, input_shape=(input_sequences.shape[1], input_sequences.shape[2])),
        Dense(128, activation='relu'),
        Dense(3)  # Output layer for pitch, velocity, time
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Fit the model
    model.fit(input_sequences, target_notes, epochs=10, batch_size=32)

    # Evaluate the model (optional)
    loss = model.evaluate(input_sequences, target_notes)
    print("Model loss:", loss)

    # Generate music using the trained model
    initial_sequence = input_sequences[0]  # Use the first input sequence as initial sequence
    generated_sequence = generate_music(model, initial_sequence, length=100)

    # Create a stream to store the generated music
    generated_stream = music21.stream.Stream()

    # Add the generated notes to the stream
    for pitch, velocity, duration in generated_sequence:
        note = music21.note.Note()
        note.pitch.ps = pitch
        note.volume.velocity = velocity
        note.duration.quarterLength = duration
        generated_stream.append(note)

    # Specify the path to save the generated MIDI file
    generated_midi_file_path = '/home/ubuntu/Deep-Learning/Disha_Kacha_DL/Project/generated_music.mid'

    # Convert the generated stream to MIDI and save it
    generated_stream.write('midi', fp=generated_midi_file_path)

    print("Generated MIDI file saved successfully!")

except Exception as e:
    print(f"Error processing MIDI file: {e}")

import music21
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Function to process MIDI file and extract note sequences
def process_midi(midi_file, sequence_length=20):
    notes = []
    for element in midi_file.flat.notes:
        notes.append((element.pitch.ps, element.volume.velocity, element.duration.quarterLength))
    note_sequences = []
    for i in range(len(notes) - sequence_length):
        sequence = notes[i:i+sequence_length]
        note_sequences.append(sequence)
    return note_sequences

# Function to generate music using the trained model
def generate_music(model, initial_sequence, length=100):
    generated_sequence = initial_sequence.copy()

    # Generate additional notes based on the initial sequence
    for _ in range(length):
        # Predict the next note based on the current sequence
        prediction = model.predict(np.array([generated_sequence]))[0]

        # Append the predicted note to the generated sequence
        generated_sequence = np.vstack([generated_sequence, prediction])

    return generated_sequence

# Load the concatenated MIDI file
concatenated_midi_file_path = '/home/ubuntu/Deep-Learning/Disha_Kacha_DL/Project/archive/concatenated_midi_albeniz.mid'

try:
    # Attempt to load the MIDI file using music21
    midi_file = music21.converter.parse(concatenated_midi_file_path)

    # Process the MIDI file to extract note sequences
    note_sequences = process_midi(midi_file)

    # Convert note sequences to numpy arrays or further processing
    note_sequences = np.array(note_sequences)

    # Check if there are any note sequences extracted
    if len(note_sequences) == 0:
        raise ValueError("No note sequences found in the MIDI file")

    # Split the data into input sequences and target notes
    input_sequences = note_sequences[:, :-1]  # All but the last note
    target_notes = note_sequences[:, -1]     # The last note

    # Reshape input sequences to match LSTM input shape
    input_sequences = input_sequences.reshape(input_sequences.shape[0], input_sequences.shape[1], 3)

    # Build the RNN model
    model = Sequential([
        LSTM(256, input_shape=(input_sequences.shape[1], input_sequences.shape[2])),
        Dense(128, activation='relu'),
        Dense(3)  # Output layer for pitch, velocity, time
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Fit the model
    model.fit(input_sequences, target_notes, epochs=10, batch_size=32)

    # Evaluate the model (optional)
    loss = model.evaluate(input_sequences, target_notes)
    print("Model loss:", loss)

    # Generate music using the trained model
    initial_sequence = input_sequences[0]  # Use the first input sequence as initial sequence
    generated_sequence = generate_music(model, initial_sequence, length=100)

    # Create a stream to store the generated music
    generated_stream = music21.stream.Stream()

    # Add the generated notes to the stream
    for pitch, velocity, duration in generated_sequence:
        note = music21.note.Note()
        note.pitch.ps = pitch
        note.volume.velocity = velocity
        note.duration.quarterLength = duration
        generated_stream.append(note)

    # Specify the path to save the generated MIDI file
    generated_midi_file_path = '/home/ubuntu/Deep-Learning/Disha_Kacha_DL/Project/generated_music.mid'

    # Convert the generated stream to MIDI and save it
    generated_stream.write('midi', fp=generated_midi_file_path)

    print("Generated MIDI file saved successfully!")

except Exception as e:
    print(f"Error processing MIDI file: {e}")