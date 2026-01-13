"""
Tests for UndoManager.
"""
import pytest

from src.core.undo_manager import UndoManager


class TestUndoManager:
    """Tests for UndoManager functionality."""
    
    def test_initial_state(self, undo_manager):
        assert not undo_manager.can_undo
        assert not undo_manager.can_redo
        assert len(undo_manager) == 0
    
    def test_push_action(self, undo_manager):
        undo_manager.push_action("Test", lambda: None, lambda: None)
        assert undo_manager.can_undo
        assert not undo_manager.can_redo
        assert len(undo_manager) == 1
    
    def test_undo_calls_undo_func(self, undo_manager):
        called = []
        undo_manager.push_action(
            "Test",
            undo_func=lambda: called.append("undo"),
            redo_func=lambda: called.append("redo")
        )
        
        undo_manager.undo()
        assert called == ["undo"]
    
    def test_redo_calls_redo_func(self, undo_manager):
        called = []
        undo_manager.push_action(
            "Test",
            undo_func=lambda: called.append("undo"),
            redo_func=lambda: called.append("redo")
        )
        
        undo_manager.undo()
        undo_manager.redo()
        assert called == ["undo", "redo"]
    
    def test_undo_moves_to_redo_stack(self, undo_manager):
        undo_manager.push_action("Test", lambda: None, lambda: None)
        assert undo_manager.can_undo
        assert not undo_manager.can_redo
        
        undo_manager.undo()
        assert not undo_manager.can_undo
        assert undo_manager.can_redo
    
    def test_new_action_clears_redo(self, undo_manager):
        undo_manager.push_action("Action 1", lambda: None, lambda: None)
        undo_manager.undo()
        assert undo_manager.can_redo
        
        undo_manager.push_action("Action 2", lambda: None, lambda: None)
        assert not undo_manager.can_redo
    
    def test_max_depth_trims_oldest(self):
        manager = UndoManager(max_depth=3)
        
        for i in range(5):
            manager.push_action(f"Action {i}", lambda: None, lambda: None)
        
        assert len(manager) == 3
    
    def test_undo_empty_returns_false(self, undo_manager):
        assert not undo_manager.undo()
    
    def test_redo_empty_returns_false(self, undo_manager):
        assert not undo_manager.redo()
    
    def test_clear(self, undo_manager):
        undo_manager.push_action("Test 1", lambda: None, lambda: None)
        undo_manager.push_action("Test 2", lambda: None, lambda: None)
        undo_manager.undo()
        
        undo_manager.clear()
        
        assert not undo_manager.can_undo
        assert not undo_manager.can_redo
        assert len(undo_manager) == 0
    
    def test_undo_description(self, undo_manager):
        undo_manager.push_action("My Action", lambda: None, lambda: None)
        assert undo_manager.undo_description == "My Action"
    
    def test_redo_description(self, undo_manager):
        undo_manager.push_action("My Action", lambda: None, lambda: None)
        undo_manager.undo()
        assert undo_manager.redo_description == "My Action"
    
    def test_multiple_undo_redo_cycle(self, undo_manager):
        values = [0]
        
        def action1_undo():
            values[0] = 0
        
        def action1_redo():
            values[0] = 1
        
        def action2_undo():
            values[0] = 1
        
        def action2_redo():
            values[0] = 2
        
        # Initial state
        values[0] = 1
        undo_manager.push_action("Set to 1", action1_undo, action1_redo)
        
        values[0] = 2
        undo_manager.push_action("Set to 2", action2_undo, action2_redo)
        
        # Undo back to 1
        undo_manager.undo()
        assert values[0] == 1
        
        # Undo back to 0
        undo_manager.undo()
        assert values[0] == 0
        
        # Redo to 1
        undo_manager.redo()
        assert values[0] == 1
        
        # Redo to 2
        undo_manager.redo()
        assert values[0] == 2
