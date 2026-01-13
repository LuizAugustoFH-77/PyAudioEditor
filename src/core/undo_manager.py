"""
Undo/Redo management for PyAudioEditor.
Implements command pattern with configurable stack depth.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import logging

from .config import UNDO_CONFIG

logger = logging.getLogger("PyAudacity")

# Type alias for undo/redo functions
ActionFunc = Callable[[], None]


@dataclass(slots=True)
class UndoAction:
    """
    Represents a single undoable action.
    
    Attributes:
        description: Human-readable description of the action
        undo_func: Function to call to undo the action
        redo_func: Function to call to redo the action
    """
    description: str
    undo_func: ActionFunc
    redo_func: ActionFunc


class UndoManager:
    """
    Manages undo/redo stacks with configurable depth.
    Uses slots for memory efficiency.
    """
    __slots__ = ('_undo_stack', '_redo_stack', '_max_depth')
    
    def __init__(self, max_depth: int = UNDO_CONFIG.max_depth) -> None:
        self._undo_stack: list[UndoAction] = []
        self._redo_stack: list[UndoAction] = []
        self._max_depth = max_depth
    
    @property
    def can_undo(self) -> bool:
        """Check if there are actions to undo."""
        return len(self._undo_stack) > 0
    
    @property
    def can_redo(self) -> bool:
        """Check if there are actions to redo."""
        return len(self._redo_stack) > 0
    
    @property
    def undo_description(self) -> Optional[str]:
        """Get description of the next action to undo."""
        if self._undo_stack:
            return self._undo_stack[-1].description
        return None
    
    @property
    def redo_description(self) -> Optional[str]:
        """Get description of the next action to redo."""
        if self._redo_stack:
            return self._redo_stack[-1].description
        return None
    
    def push_action(
        self, 
        description: str, 
        undo_func: ActionFunc, 
        redo_func: ActionFunc
    ) -> None:
        """
        Push a new undoable action onto the stack.
        Clears the redo stack when a new action is pushed.
        """
        action = UndoAction(description, undo_func, redo_func)
        self._undo_stack.append(action)
        
        # Trim stack if over max depth
        if len(self._undo_stack) > self._max_depth:
            self._undo_stack.pop(0)
        
        # Clear redo stack on new action
        self._redo_stack.clear()
        
        logger.debug("Undo action pushed: %s", description)
    
    def undo(self) -> bool:
        """
        Undo the last action.
        Returns True if successful, False if nothing to undo.
        """
        if not self._undo_stack:
            logger.debug("Nothing to undo")
            return False
        
        action = self._undo_stack.pop()
        try:
            action.undo_func()
            self._redo_stack.append(action)
            logger.info("Undo: %s", action.description)
            return True
        except Exception as e:
            logger.error("Error during undo '%s': %s", action.description, e, exc_info=True)
            return False
    
    def redo(self) -> bool:
        """
        Redo the last undone action.
        Returns True if successful, False if nothing to redo.
        """
        if not self._redo_stack:
            logger.debug("Nothing to redo")
            return False
        
        action = self._redo_stack.pop()
        try:
            action.redo_func()
            self._undo_stack.append(action)
            logger.info("Redo: %s", action.description)
            return True
        except Exception as e:
            logger.error("Error during redo '%s': %s", action.description, e, exc_info=True)
            return False
    
    def clear(self) -> None:
        """Clear both undo and redo stacks."""
        self._undo_stack.clear()
        self._redo_stack.clear()
        logger.debug("Undo/Redo stacks cleared")
    
    def __len__(self) -> int:
        """Return number of undoable actions."""
        return len(self._undo_stack)
    
    def __repr__(self) -> str:
        return f"UndoManager(undo={len(self._undo_stack)}, redo={len(self._redo_stack)})"
