# DATS_6303_10_DL_Project

Welcome to our Classical Music Generation application! This project leverages Long Short-Term Memory (LSTM) neural networks with embedding layers to generate classical music sequences. Using MIDI files as input, the application extracts detailed note sequences, including pitch, velocity, start, and end times. These sequences are then used to train the LSTM model, which learns patterns in the music data. Once trained, the model can generate new music sequences based on a provided seed sequence. By adjusting parameters such as temperature, users can control the variability and creativity of the generated music. The generated music sequences are then converted to MIDI format to ensure all notes are audible. With this application, users can explore the fascinating world of classical music generation, creating unique compositions with just a click of a button!

## How to run the app and code files?

Open the main folder, you will find 3 files
- act_vs_pred.py
- Project_Streamlit.py
- Music with Tempo Categorical.py
- requirements.txt

Basic requirements based on our AWS image and server are provided in requirements.txt


1. ```pip install requirements.txt```

Download the dataset that is in the zip file, And save it ou your instance.

2. Open the ```act_vs_pred.py```
  
- Change the directory path of the dataset as stored in your system and include the composer name on which you want your model to be trained. (line 74)
- Change the path where you want the predicted midi file to be generated. (line 194)
- Same way change the path for actual music. (line 207)

  ```python3 act_vs_pred.py```


3. Open the ```Music with Tempo Categorical.py```

This code will get you a predicted music based on tempo changes as in, 2 or 3 notes being played in at the same time.

- Change the directory path of the dataset as stored in your system and include the composer name on which you want your model to be trained. (line 138)
- Change the path for music to be saved. (line 290)

```python3 Music with Tempo Categorical.py```




