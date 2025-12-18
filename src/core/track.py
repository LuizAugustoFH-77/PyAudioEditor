import numpy as np

class AudioTrack:
    """
    Represents a single audio track with its data, metadata, and state.
    """
    def __init__(self, name="Track"):
        self.name = name
        self.data = None # numpy array (samples, channels)
        self.gain = 1.0
        self.pan = 0.0 # -1.0 to 1.0
        self.muted = False
        self.soloed = False
        self.samplerate = 44100
        self.splits = [] # List of sample indices where the track is split

    def set_data(self, data, samplerate):
        """Sets the audio data and samplerate for the track."""
        self.data = data
        self.samplerate = samplerate
        return self # Fluent API
        
    def get_duration_samples(self):
        """Returns the total number of samples in the track."""
        return len(self.data) if self.data is not None else 0

    def get_duration_seconds(self):
        """Returns the duration of the track in seconds."""
        return self.get_duration_samples() / self.samplerate if self.samplerate > 0 else 0
