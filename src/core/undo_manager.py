import numpy as np
from src.utils.logger import logger

class UndoAction:
    def __init__(self, description, undo_func, redo_func):
        self.description = description
        self.undo_func = undo_func
        self.redo_func = redo_func

class UndoManager:
    def __init__(self, max_depth=20):
        self.undo_stack = []
        self.redo_stack = []
        self.max_depth = max_depth

    def push_action(self, description, undo_func, redo_func):
        action = UndoAction(description, undo_func, redo_func)
        self.undo_stack.append(action)
        if len(self.undo_stack) > self.max_depth:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        logger.debug(f"Undo action pushed: {description}")

    def undo(self):
        if not self.undo_stack:
            logger.debug("Nothing to undo")
            return False
        
        action = self.undo_stack.pop()
        try:
            action.undo_func()
            self.redo_stack.append(action)
            logger.info(f"Undo: {action.description}")
            return True
        except Exception as e:
            logger.error(f"Error during undo: {e}")
            return False

    def redo(self):
        if not self.redo_stack:
            logger.debug("Nothing to redo")
            return False
        
        action = self.redo_stack.pop()
        try:
            action.redo_func()
            self.undo_stack.append(action)
            logger.info(f"Redo: {action.description}")
            return True
        except Exception as e:
            logger.error(f"Error during redo: {e}")
            return False

    def clear(self):
        self.undo_stack.clear()
        self.redo_stack.clear()
        logger.debug("Undo/Redo stacks cleared")
