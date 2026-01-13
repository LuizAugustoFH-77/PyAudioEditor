"""
Tests for Project.
"""
import pytest
import numpy as np

from src.core.project import Project
from src.core.track import AudioTrack
from src.core.config import AUDIO_CONFIG


class TestProject:
    """Tests for Project functionality."""
    
    def test_default_initialization(self):
        project = Project()
        assert project.name == "Untitled Project"
        assert project.samplerate == AUDIO_CONFIG.default_samplerate
        assert project.tracks == []
        assert project.clipboard is None
    
    def test_custom_initialization(self):
        project = Project(name="My Project", samplerate=48000)
        assert project.name == "My Project"
        assert project.samplerate == 48000
    
    def test_add_track(self, empty_project, sample_track):
        index = empty_project.add_track(sample_track)
        assert index == 0
        assert len(empty_project.tracks) == 1
        assert empty_project.tracks[0] is sample_track
    
    def test_remove_track(self, project_with_track):
        track = project_with_track.remove_track(0)
        assert track is not None
        assert len(project_with_track.tracks) == 0
    
    def test_remove_track_invalid_index(self, project_with_track):
        track = project_with_track.remove_track(999)
        assert track is None
        assert len(project_with_track.tracks) == 1
    
    def test_get_track(self, project_with_track):
        track = project_with_track.get_track(0)
        assert track is not None
        assert track.name == "Test Track"
    
    def test_get_track_invalid_index(self, project_with_track):
        assert project_with_track.get_track(999) is None
        assert project_with_track.get_track(-1) is None
    
    def test_duration_samples(self, project_with_track):
        expected = AUDIO_CONFIG.default_samplerate
        assert project_with_track.duration_samples == expected
    
    def test_duration_seconds(self, project_with_track):
        assert np.isclose(project_with_track.duration_seconds, 1.0, atol=0.01)
    
    def test_duration_empty_project(self, empty_project):
        assert empty_project.duration_samples == 0
        assert empty_project.duration_seconds == 0.0
    
    def test_has_solo(self, project_with_track):
        assert not project_with_track.has_solo
        project_with_track.tracks[0].soloed = True
        assert project_with_track.has_solo
    
    def test_get_active_tracks_all(self, project_with_track):
        active = project_with_track.get_active_tracks()
        assert len(active) == 1
    
    def test_get_active_tracks_muted(self, project_with_track):
        project_with_track.tracks[0].muted = True
        active = project_with_track.get_active_tracks()
        assert len(active) == 0
    
    def test_get_active_tracks_solo(self, sample_stereo_audio):
        project = Project()
        track1 = AudioTrack(name="Track 1")
        track1.set_data(sample_stereo_audio, 44100)
        track2 = AudioTrack(name="Track 2")
        track2.set_data(sample_stereo_audio, 44100)
        
        project.add_track(track1)
        project.add_track(track2)
        
        track1.soloed = True
        active = project.get_active_tracks()
        
        assert len(active) == 1
        assert active[0] is track1
    
    def test_mix_down(self, project_with_track):
        mixed = project_with_track.mix_down()
        assert mixed is not None
        assert mixed.shape == (project_with_track.duration_samples, 2)
    
    def test_mix_down_empty(self, empty_project):
        assert empty_project.mix_down() is None
    
    def test_mix_down_respects_gain(self, sample_stereo_audio):
        project = Project()
        track = AudioTrack(name="Test", gain=0.5)
        track.set_data(sample_stereo_audio.copy(), 44100)
        project.add_track(track)
        
        mixed = project.mix_down()
        expected = sample_stereo_audio * 0.5
        
        assert np.allclose(mixed, expected, atol=1e-5)
    
    def test_clear(self, project_with_track):
        project_with_track.clipboard = {"test": "data"}
        project_with_track.clear()
        
        assert len(project_with_track.tracks) == 0
        assert project_with_track.clipboard is None
    
    def test_undo_manager_lazy_init(self, empty_project):
        manager = empty_project.undo_manager
        assert manager is not None
