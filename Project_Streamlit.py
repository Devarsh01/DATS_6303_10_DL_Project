import streamlit as st
import numpy as np
import pretty_midi
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import io
import fluidsynth
from scipy.io import wavfile
import subprocess
from pydub import AudioSegment
from midi2audio import FluidSynth
import mido

# Check for GPU availability
if tf.config.experimental.list_physical_devices('GPU'):
    st.write("Using GPU")
else:
    st.write("Using CPU")


@st.cache_resource
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


def create_input_target_sequences(sequence, seq_length, vocab_size):
    """Create input and target sequences from detailed note sequences."""
    input_sequences = []
    target_sequences = []
    for i in range(len(sequence) - seq_length):
        input_seq = [[event['pitch']] for event in sequence[i:i + seq_length]]
        target_pitch = sequence[i + seq_length]['pitch']
        target_seq = to_categorical(target_pitch, num_classes=vocab_size)
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)
    return np.array(input_sequences), np.array(target_sequences)


def load_existing_model(model_path):
    """Load an existing TensorFlow model from the given path."""
    return load_model(model_path)


def generate_music(model, seed_sequence, length=10, steps_per_second=5, temperature=2):
    """Generate music from a seed sequence aiming for a total duration using a temperature parameter."""
    generated_sequence = np.copy(seed_sequence)  # Copy to avoid modifying the original seed
    total_steps = length * steps_per_second  # Total steps needed for desired duration

    for _ in range(total_steps):
        # Predict the next step using the last 'seq_length' elements in the generated_sequence
        prediction = model.predict(np.expand_dims(generated_sequence[-seq_length:], axis=0))[0]

        # Apply temperature to the prediction probabilities and normalize
        prediction = np.log(prediction + 1e-8) / temperature  # Smoothing and apply temperature
        exp_prediction = np.exp(prediction)
        prediction = exp_prediction / np.sum(exp_prediction)

        # Sample an index from the probability array
        predicted_pitch = np.random.choice(len(prediction), p=prediction)

        # Append the predicted pitch to the generated sequence
        generated_sequence = np.vstack([generated_sequence, predicted_pitch])

    st.write("Generated sequence with variability:", generated_sequence[-30:])
    return generated_sequence


def generated_to_midi(generated_sequence, fs=100, total_duration=6):
    """Convert generated sequence to MIDI file, ensuring all notes are audible."""
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    # Calculate the duration per step based on total duration and number of notes
    duration_per_step = total_duration / len(generated_sequence)
    min_duration = 0.1  # Set a minimum note duration for clarity

    # Initialize the current time to track the start time of each note
    current_time = 0

    for step in generated_sequence:
        pitch = int(np.clip(step[0], 21, 108))  # Scale and clip pitch values to MIDI range
        velocity = 100  # Fixed velocity for all notes

        # Set start and end time for each note
        start = current_time
        end = start + max(min_duration, duration_per_step)

        # Create a MIDI note with the determined pitch, velocity, start, and end times
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start,
            end=end
        )
        instrument.notes.append(note)

        # Update the current time to the end of this note for the next note
        current_time = end

    # Add the instrument to the PrettyMIDI object
    pm.instruments.append(instrument)
    print("The generated MIDI")
    return pm

def midi_to_audio(midi_data, output_path='output.wav'):
    # Write the MIDI data to a temporary file
    temp_midi = 'temp.mid'
    with open(temp_midi, 'wb') as midi_file:
        midi_file.write(midi_data.getvalue())

    # Convert MIDI to WAV using FluidSynth
    subprocess.run(['fluidsynth', '-F', output_path, '-r', '44100', '-g', '1.0', soundfont_path, temp_midi])

    # Load the WAV file for further use (optional)
    return AudioSegment.from_wav(output_path)

# Define the directory options and their corresponding model paths
composer_models = {
    'albeniz': '/path/to/albeniz_model.h5',
    'bach': '/Users/dishakacha/Downloads/Deep_Learning/Deep_Learing/Project/Model/model_bach.h5',
    'balakirew': '/path/to/balakirew_model.h5'
}

# Define the directory options for training and prediction MIDI files
training_midi_directory_options = {
    'albeniz': '/Users/dishakacha/Downloads/Deep_Learning/Deep_Learing/Project/archive/albeniz',
    'bach': '/Users/dishakacha/Downloads/Deep_Learning/Deep_Learing/Project/archive/bach',
    'balakir': '/Users/dishakacha/Downloads/Deep_Learning/Deep_Learing/Project/archive/balakir'
}

prediction_midi_directory_options = {
    'albeniz': '/Users/dishakacha/Downloads/Deep_Learning/Deep_Learing/Project/archive/albeniz',
    'bach': '/Users/dishakacha/Downloads/Deep_Learning/Deep_Learing/Project/archive/bach',
    'balakir': '/Users/dishakacha/Downloads/Deep_Learning/Deep_Learing/Project/archive/balakir'
}

st.title("Classical Music Generation")

# Select composer for training
selected_composer_training = st.selectbox("Select composer for training:", list(composer_models.keys()))
training_directory_path = training_midi_directory_options[selected_composer_training]

# Select composer for prediction
selected_composer_prediction = st.selectbox("Select composer for prediction:", list(composer_models.keys()))
prediction_directory_path = prediction_midi_directory_options[selected_composer_prediction]

# Parameters for the sequence creation
seq_length = 30  # Length of the input sequences
vocab_size = 128  # Number of unique pitches (for MIDI, typically 128)

if st.button('Load Data'):
    # Load training data
    training_sequences = load_midi_details(training_directory_path)
    training_input_sequences, training_target_sequences = create_input_target_sequences(training_sequences, seq_length,
                                                                                        vocab_size)
    st.write(f"Loaded {len(training_sequences)} training sequences.")
    st.write(f"Training input sequences shape: {training_input_sequences.shape}")
    st.write(f"Training target sequences shape: {training_target_sequences.shape}")

    # Load the selected model for training
    model_path = composer_models[selected_composer_training]
    try:
        training_model = load_existing_model(model_path)
        st.success("Training model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading training model: {e}")


# Generate music based on the loaded model
if st.button('Generate Music'):

    # Load prediction data
    prediction_sequences = load_midi_details(prediction_directory_path)
    prediction_input_sequences, prediction_target_sequences = create_input_target_sequences(prediction_sequences,
                                                                                            seq_length, vocab_size)
    st.write(f"Loaded {len(prediction_sequences)} prediction sequences.")
    st.write(f"Prediction input sequences shape: {prediction_input_sequences.shape}")
    st.write(f"Prediction target sequences shape: {prediction_target_sequences.shape}")

    training_model = load_existing_model(composer_models[selected_composer_training])

    seed_index = 0
    seed_sequence = prediction_input_sequences[seed_index]
    # Generate music using the training model
    generated_music = generate_music(training_model, seed_sequence)

    # Convert generated sequence to MIDI
    generated_music_midi = generated_to_midi(generated_music)

    # Save the generated MIDI to a temporary file
    with io.BytesIO() as buffer:
        generated_music_midi.write(buffer)
        buffer.seek(0)
        audio = midi_to_audio(buffer)
        st.audio(audio.raw_data, format='audio/wav')