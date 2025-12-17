This is the Python script for training the Music Director Classifier's CNN model using TensorFlow and librosa.
It segments audio files into 3-second clips, extracts Mel-Spectrograms, and applies StandardScaler for crucial feature normalization.
The script defines a deep 2D CNN architecture, trains it with callbacks (EarlyStopping, ReduceLROnPlateau), and evaluates performance using loss, accuracy, and a confusion matrix plot.
