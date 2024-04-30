#%%
import numpy as np
import pretty_midi
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
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

# Load model
def load_existing_model(model_path):
    return load_model(model_path)

# Generate music function as previously defined
def generate_music(model, seed_sequence, length=10, steps_per_second=5, temperature=1):
    """Generate music from a seed sequence aiming for a total duration using a temperature parameter."""
    generated_sequence = np.copy(seed_sequence)  # Copy to avoid modifying the original seed
    total_steps = length * steps_per_second  # Total steps needed for desired duration

    for _ in range(total_steps):
        prediction = model.predict(np.expand_dims(generated_sequence[-seq_length:], axis=0))[0]
        prediction = np.log(prediction + 1e-8) / temperature
        exp_prediction = np.exp(prediction)
        prediction = exp_prediction / np.sum(exp_prediction)
        predicted_pitch = np.random.choice(len(prediction), p=prediction)
        generated_sequence = np.vstack([generated_sequence, predicted_pitch])

    return generated_sequence

# Convert generated sequence to MIDI as previously defined
def generated_to_midi(generated_sequence, fs=100, total_duration=6):
    """Convert generated sequence to MIDI file, ensuring all notes are audible."""
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    duration_per_step = total_duration / len(generated_sequence)
    min_duration = 0.1
    current_time = 0

    for step in generated_sequence:
        pitch = int(np.clip(step[0], 21, 108))
        velocity = 100
        start = current_time
        end = start + max(min_duration, duration_per_step)
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start,
            end=end
        )
        instrument.notes.append(note)
        current_time = end

    pm.instruments.append(instrument)
    return pm

# Path to the model
model_path = '/path/to/your/model.h5'
model = load_existing_model(model_path)

# Directory containing new MIDI files for generating music
new_midi_directory = '/path/to/new/midi/files/'
sequences = load_midi_details(new_midi_directory)
seq_length = 30  # As used earlier
vocab_size = 128  # As used earlier
input_sequences, target_sequences = create_input_target_sequences(sequences, seq_length, vocab_size)

# Assuming you have a specific input sequence you want to start generating from
seed_index = 1000  # example index
if len(input_sequences) > seed_index:
    seed_sequence = input_sequences[seed_index]
    generated_music = generate_music(model, seed_sequence, length=20, steps_per_second=2)
    generated_music_midi = generated_to_midi(generated_music, total_duration=10)
    output_path = '/path/where/you/want/to/save/generated_music.mid'
    generated_music_midi.write(output_path)
else:
    print("Not enough input sequences available.")
