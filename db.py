import sqlite3
from contextlib import contextmanager

DB_PATH = "audiobook.db"


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Creates tables if missing, and adds new columns to older databases if needed."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS books (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                total_pages INTEGER,
                detection_method TEXT,
                uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chapters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id INTEGER NOT NULL,
                chapter_index INTEGER NOT NULL,
                title TEXT,
                start_page INTEGER,
                end_page INTEGER,
                paragraph_count INTEGER,
                content TEXT,
                FOREIGN KEY (book_id) REFERENCES books (id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id INTEGER NOT NULL,
                chapter_index INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                char_count INTEGER,
                FOREIGN KEY (book_id) REFERENCES books (id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS playback_progress (
                book_id INTEGER PRIMARY KEY,
                chapter_index INTEGER,
                chunk_index INTEGER,
                position_seconds REAL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Migration: if this database was created before Phase 6, add the new
        # columns it needs without losing any data already saved.
        existing_columns = {row["name"] for row in conn.execute("PRAGMA table_info(chapters)").fetchall()}
        if "approved" not in existing_columns:
            conn.execute("ALTER TABLE chapters ADD COLUMN approved INTEGER DEFAULT 0")
        if "flagged" not in existing_columns:
            conn.execute("ALTER TABLE chapters ADD COLUMN flagged INTEGER DEFAULT 0")
        if "flag_reason" not in existing_columns:
            conn.execute("ALTER TABLE chapters ADD COLUMN flag_reason TEXT")

        # Migration: add generation-tracking columns to chunks (added in Phase 9)
        chunk_columns = {row["name"] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
        if "status" not in chunk_columns:
            conn.execute("ALTER TABLE chunks ADD COLUMN status TEXT DEFAULT 'pending'")
        if "audio_path" not in chunk_columns:
            conn.execute("ALTER TABLE chunks ADD COLUMN audio_path TEXT")
        if "voice" not in chunk_columns:
            conn.execute("ALTER TABLE chunks ADD COLUMN voice TEXT")
        if "error_message" not in chunk_columns:
            conn.execute("ALTER TABLE chunks ADD COLUMN error_message TEXT")


def save_book(filename, total_pages, detection_method, chapters):
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO books (filename, total_pages, detection_method) VALUES (?, ?, ?)",
            (filename, total_pages, detection_method),
        )
        book_id = cursor.lastrowid

        for index, chap in enumerate(chapters):
            conn.execute(
                """INSERT INTO chapters
                   (book_id, chapter_index, title, start_page, end_page,
                    paragraph_count, content, flagged, flag_reason, approved)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
                (
                    book_id,
                    index,
                    chap["title"],
                    chap["start_page"],
                    chap["end_page"],
                    chap["paragraph_count"],
                    chap["content"],
                    1 if chap.get("flagged") else 0,
                    chap.get("flag_reason"),
                ),
            )
        return book_id


def get_all_books():
    """Returns a summary list of every book, including audio generation progress."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT b.id, b.filename, b.total_pages, b.detection_method, b.uploaded_at,
                   COUNT(c.id) as total_chunks,
                   SUM(CASE WHEN c.status = 'completed' THEN 1 ELSE 0 END) as completed_chunks
            FROM books b
            LEFT JOIN chunks c ON c.book_id = b.id
            GROUP BY b.id
            ORDER BY b.uploaded_at DESC
        """).fetchall()
        books = [dict(row) for row in rows]
        for book in books:
            book["completed_chunks"] = book["completed_chunks"] or 0
        return books


def delete_book(book_id):
    """Removes a book and everything associated with it from the database."""
    with get_connection() as conn:
        conn.execute("DELETE FROM chunks WHERE book_id = ?", (book_id,))
        conn.execute("DELETE FROM chapters WHERE book_id = ?", (book_id,))
        conn.execute("DELETE FROM playback_progress WHERE book_id = ?", (book_id,))
        conn.execute("DELETE FROM books WHERE id = ?", (book_id,))


def get_book_with_chapters(book_id):
    with get_connection() as conn:
        book_row = conn.execute("SELECT * FROM books WHERE id = ?", (book_id,)).fetchone()
        if not book_row:
            return None
        chapter_rows = conn.execute(
            "SELECT * FROM chapters WHERE book_id = ? ORDER BY chapter_index",
            (book_id,),
        ).fetchall()
        return {
            "book": dict(book_row),
            "chapters": [dict(row) for row in chapter_rows],
        }


def set_chapter_approved(book_id, chapter_index, approved: bool):
    with get_connection() as conn:
        conn.execute(
            "UPDATE chapters SET approved = ? WHERE book_id = ? AND chapter_index = ?",
            (1 if approved else 0, book_id, chapter_index),
        )


def save_chunks(book_id, chapter_index, chunks):
    """
    Replaces any existing chunks for this chapter with a fresh set.
    'chunks' is a list of text strings, in reading order.
    """
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM chunks WHERE book_id = ? AND chapter_index = ?",
            (book_id, chapter_index),
        )
        for index, text in enumerate(chunks):
            conn.execute(
                "INSERT INTO chunks (book_id, chapter_index, chunk_index, text, char_count) "
                "VALUES (?, ?, ?, ?, ?)",
                (book_id, chapter_index, index, text, len(text)),
            )


def get_chunks(book_id, chapter_index):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM chunks WHERE book_id = ? AND chapter_index = ? ORDER BY chunk_index",
            (book_id, chapter_index),
        ).fetchall()
        return [dict(row) for row in rows]


def set_chunk_status(book_id, chapter_index, chunk_index, status, audio_path=None, voice=None, error_message=None):
    """
    Updates one chunk's generation status.
    audio_path/voice are only overwritten if a new value is given (so re-marking
    a chunk 'processing' doesn't wipe out a previously saved audio path).
    error_message is always set as given, so a success properly clears an old error.
    """
    with get_connection() as conn:
        conn.execute(
            """UPDATE chunks
               SET status = ?,
                   audio_path = COALESCE(?, audio_path),
                   voice = COALESCE(?, voice),
                   error_message = ?
               WHERE book_id = ? AND chapter_index = ? AND chunk_index = ?""",
            (status, audio_path, voice, error_message, book_id, chapter_index, chunk_index),
        )


def reset_stuck_chunks():
    """
    Called once when the app starts up. Any chunk still marked 'processing'
    is leftover from a previous run that was interrupted (crash, Ctrl+C,
    power loss, etc) — since the app just started fresh, nothing can
    actually still be running, so we reset these back to 'pending' so a
    future 'Generate' click will retry them instead of them being stuck
    showing a lie forever.
    Returns how many chunks were reset, so the startup log can report it.
    """
    with get_connection() as conn:
        cursor = conn.execute("UPDATE chunks SET status = 'pending' WHERE status = 'processing'")
        return cursor.rowcount


def save_playback_progress(book_id, chapter_index, chunk_index, position_seconds):
    """Saves (or updates) the single 'last listened to' position for a book."""
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO playback_progress (book_id, chapter_index, chunk_index, position_seconds, updated_at)
               VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(book_id) DO UPDATE SET
                   chapter_index = excluded.chapter_index,
                   chunk_index = excluded.chunk_index,
                   position_seconds = excluded.position_seconds,
                   updated_at = excluded.updated_at""",
            (book_id, chapter_index, chunk_index, position_seconds),
        )


def get_playback_progress(book_id):
    with get_connection() as conn:
        row = conn.execute(
            """SELECT p.*, c.title AS chapter_title
               FROM playback_progress p
               LEFT JOIN chapters c ON c.book_id = p.book_id AND c.chapter_index = p.chapter_index
               WHERE p.book_id = ?""",
            (book_id,),
        ).fetchone()
        return dict(row) if row else None


def reset_chunks_from(book_id, chapter_index, from_chunk_index):
    """
    Forces every chunk from from_chunk_index onward back to 'pending',
    even if it was already 'completed' — used when the user wants to
    regenerate a chapter's remaining audio with a different voice.
    """
    with get_connection() as conn:
        conn.execute(
            """UPDATE chunks
               SET status = 'pending', audio_path = NULL, voice = NULL, error_message = NULL
               WHERE book_id = ? AND chapter_index = ? AND chunk_index >= ?""",
            (book_id, chapter_index, from_chunk_index),
        )