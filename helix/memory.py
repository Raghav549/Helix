"""Opt-in, editable local conversation memory (not encrypted at rest)."""
from __future__ import annotations
import os
from pathlib import Path
import sqlite3
import threading
import time


class MemoryStore:
    def __init__(self, path: str = 'out/memory.sqlite3'):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, created_at INTEGER NOT NULL)')
        os.chmod(self.path, 0o600)

    def _connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def list(self):
        with self._lock, self._connect() as conn:
            rows = conn.execute('SELECT id, text, created_at FROM memories ORDER BY id DESC LIMIT 100').fetchall()
        return [{'id': row[0], 'text': row[1], 'created_at': row[2]} for row in rows]

    def add(self, text: str, *, consent: bool = False):
        if consent is not True:
            raise PermissionError('explicit consent is required to save memory')
        self._validate(text)
        with self._lock, self._connect() as conn:
            cur = conn.execute('INSERT INTO memories (text, created_at) VALUES (?, ?)', (text.strip(), int(time.time())))
            return cur.lastrowid

    def update(self, memory_id: int, text: str, *, consent: bool = False):
        if consent is not True: raise PermissionError('explicit consent is required to edit memory')
        self._validate(text)
        with self._lock, self._connect() as conn:
            return conn.execute('UPDATE memories SET text = ? WHERE id = ?', (text.strip(), memory_id)).rowcount > 0

    def delete(self, memory_id: int):
        with self._lock, self._connect() as conn:
            return conn.execute('DELETE FROM memories WHERE id = ?', (memory_id,)).rowcount > 0

    def clear(self):
        with self._lock, self._connect() as conn:
            return conn.execute('DELETE FROM memories').rowcount

    @staticmethod
    def _validate(text):
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 1000:
            raise ValueError('memory must contain 1..1000 characters')
