import os
import re
import shutil
from collections import Counter

import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles

import db
import tts

app = FastAPI()

UPLOAD_DIR = "uploads"
AUDIO_DIR = "audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
db.init_db()

# Startup recovery: clean up anything left over from a previous crash or forced shutdown.
_reset_count = db.reset_stuck_chunks()
if _reset_count:
    print(f"[startup] Recovered {_reset_count} chunk(s) that were stuck 'processing' from a previous run.")

_removed_temp_files = 0
for root, _, files in os.walk(AUDIO_DIR):
    for name in files:
        if name.endswith(".tmp"):
            os.remove(os.path.join(root, name))
            _removed_temp_files += 1
if _removed_temp_files:
    print(f"[startup] Removed {_removed_temp_files} leftover temporary audio file(s).")

MAX_FILE_SIZE_MB = 50

LEGAL_KEYWORDS = [
    "all rights reserved",
    "copyright ©",
    "copyright (c)",
    "isbn",
    "no part of this publication",
    "printed in the united states",
    "library of congress",
    "trademark",
    "publisher's note",
    "penguin random house",
]

CHAPTER_PATTERN = re.compile(r"^(chapter|part)\s+([0-9ivxlcdm]+)\b", re.IGNORECASE)

NON_CHAPTER_TITLES = {
    "title page", "copyright", "epigraph", "dedication", "contents",
    "table of contents", "acknowledgments", "acknowledgements", "notes",
    "index", "about the author", "also by", "foreword", "preface",
    "praise for", "cover", "half title", "colophon",
}


def is_probably_not_a_chapter(title: str, page_count: int) -> bool:
    normalized = title.strip().lower()
    if normalized in NON_CHAPTER_TITLES:
        return True
    if page_count <= 2 and not CHAPTER_PATTERN.match(title.strip()):
        return True
    return False


def is_page_number_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.isdigit():
        return True
    if re.fullmatch(r"[ivxlcdm]+", stripped.lower()) and len(stripped) <= 6:
        return True
    return False


def is_legal_boilerplate(line: str) -> bool:
    lower = line.lower()
    return any(keyword in lower for keyword in LEGAL_KEYWORDS)


def clean_pages(pages_text: list[str]) -> str:
    pages_lines = [page.split("\n") for page in pages_text]

    line_counter = Counter()
    for lines in pages_lines:
        seen_on_this_page = set()
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) < 60:
                seen_on_this_page.add(stripped)
        for line in seen_on_this_page:
            line_counter[line] += 1

    num_pages = len(pages_lines)
    threshold = max(2, int(num_pages * 0.3))
    repeated_lines = {line for line, count in line_counter.items() if count >= threshold}

    cleaned_pages = []
    for lines in pages_lines:
        kept_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped in repeated_lines:
                continue
            if is_page_number_line(stripped):
                continue
            if is_legal_boilerplate(stripped):
                continue
            kept_lines.append(stripped)
        if kept_lines:
            cleaned_pages.append("\n".join(kept_lines))

    return "\n\n".join(cleaned_pages)


def detect_chapters_from_toc(doc, num_pages):
    toc = doc.get_toc()
    if not toc:
        return None

    top_level = [entry for entry in toc if entry[0] == 1] or toc

    chapters = []
    for i, (level, title, page) in enumerate(top_level):
        start_page = max(0, page - 1)
        if i + 1 < len(top_level):
            next_start = max(0, top_level[i + 1][2] - 1)
            end_page = max(start_page, next_start - 1)
        else:
            end_page = num_pages - 1
        chapters.append({"title": title.strip(), "start_page": start_page, "end_page": end_page})
    return chapters


def detect_chapters_heuristic(pages_text):
    chapter_starts = []
    for page_index, page_text in enumerate(pages_text):
        for line in page_text.split("\n"):
            stripped = line.strip()
            if len(stripped) > 60:
                continue
            if CHAPTER_PATTERN.match(stripped):
                chapter_starts.append((page_index, stripped))
                break

    if not chapter_starts:
        return None

    chapters = []
    for i, (start_page, title) in enumerate(chapter_starts):
        if i + 1 < len(chapter_starts):
            end_page = chapter_starts[i + 1][0] - 1
        else:
            end_page = len(pages_text) - 1
        chapters.append({"title": title, "start_page": start_page, "end_page": end_page})
    return chapters


def split_into_sentences(text: str):
    """
    A simple sentence splitter: breaks text after '.', '!', or '?' when followed
    by whitespace and a capital letter or quote mark. Not perfect (things like
    "Dr. Smith" can trip it up), but good enough for chunking purposes.
    """
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\u201c])', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(full_text: str, max_chars: int = 500):
    """
    Splits a chapter's full text into TTS-sized chunks.
    Sentences are never split apart mid-way — a chunk just stops growing
    once the next sentence would push it past max_chars.
    """
    paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        for sentence in split_into_sentences(paragraph):
            candidate = (current_chunk + " " + sentence).strip() if current_chunk else sentence
            if len(candidate) <= max_chars or not current_chunk:
                current_chunk = candidate
            else:
                chunks.append(current_chunk)
                current_chunk = sentence

        # Prefer to end a chunk at a paragraph boundary once it's reasonably full
        if current_chunk and len(current_chunk) >= max_chars * 0.6:
            chunks.append(current_chunk)
            current_chunk = ""

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def detect_extraction_problems(content: str, page_count: int):
    """
    Basic sanity checks on a chapter's extracted text.
    Returns a human-readable reason if something looks off, or None if it looks fine.
    """
    if not content.strip():
        return "No readable text was found in this chapter."

    chars_per_page = len(content) / page_count if page_count > 0 else 0
    if chars_per_page < 100:
        return "Very little text was extracted — this chapter may be mostly images, or there may be an extraction problem."

    letters_and_spaces = sum(1 for c in content if c.isalpha() or c.isspace())
    ratio = letters_and_spaces / len(content) if content else 0
    if ratio < 0.7:
        return "This chapter's text looks unusual (a lot of non-letter characters) — it may have extracted incorrectly."

    return None


def build_book_structure(doc, pages_text):
    num_pages = len(pages_text)

    chapters_meta = detect_chapters_from_toc(doc, num_pages)
    method = "toc"

    if not chapters_meta:
        chapters_meta = detect_chapters_heuristic(pages_text)
        method = "heuristic"

    if not chapters_meta:
        chapters_meta = [{"title": "Full Book (no chapters detected)", "start_page": 0, "end_page": num_pages - 1}]
        method = "none"

    if method == "toc":
        filtered = []
        for chap in chapters_meta:
            page_count = (chap["end_page"] - chap["start_page"]) + 1
            if is_probably_not_a_chapter(chap["title"], page_count):
                continue
            filtered.append(chap)
        if filtered:
            chapters_meta = filtered

    chapters = []
    for chap in chapters_meta:
        chapter_pages = pages_text[chap["start_page"]: chap["end_page"] + 1]
        cleaned_text = clean_pages(chapter_pages)
        paragraphs = [p.strip() for p in cleaned_text.split("\n\n") if p.strip()]
        page_count = (chap["end_page"] - chap["start_page"]) + 1
        flag_reason = detect_extraction_problems(cleaned_text, page_count)

        chapters.append({
            "title": chap["title"],
            "start_page": chap["start_page"] + 1,
            "end_page": chap["end_page"] + 1,
            "paragraph_count": len(paragraphs),
            "content": cleaned_text,
            "preview": cleaned_text[:250],
            "flagged": flag_reason is not None,
            "flag_reason": flag_reason,
        })

    return chapters, method


@app.get("/api/hello")
def hello():
    return {"message": "Backend is alive and working!"}


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    contents = await file.read()

    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f} MB). Max allowed is {MAX_FILE_SIZE_MB} MB.",
        )

    if not contents.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="This file doesn't look like a valid PDF.")

    safe_filename = os.path.basename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, safe_filename)
    with open(save_path, "wb") as f:
        f.write(contents)

    try:
        doc = fitz.open(save_path)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open this PDF. It may be corrupt.")

    num_pages = len(doc)
    pages_text = [page.get_text() for page in doc]

    raw_text = "\n".join(pages_text)
    avg_chars_per_page = len(raw_text) / num_pages if num_pages > 0 else 0
    likely_scanned = avg_chars_per_page < 50

    if likely_scanned:
        doc.close()
        return {
            "filename": safe_filename,
            "pages": num_pages,
            "likely_scanned": True,
            "message": "This looks like a scanned PDF. OCR isn't built yet, so text extraction may be empty or unreliable.",
            "chapters": [],
            "detection_method": "none",
        }

    chapters, method = build_book_structure(doc, pages_text)
    doc.close()

    # Save the book and its chapters to the database so it survives a restart.
    book_id = db.save_book(
        filename=safe_filename,
        total_pages=num_pages,
        detection_method=method,
        chapters=chapters,
    )

    # Don't send the full chapter text back over the network here — just previews.
    response_chapters = [
        {k: v for k, v in chap.items() if k != "content"}
        for chap in chapters
    ]

    return {
        "book_id": book_id,
        "filename": safe_filename,
        "pages": num_pages,
        "likely_scanned": False,
        "message": f"Extracted, cleaned, structured into {len(chapters)} chapter(s), and saved.",
        "chapters": response_chapters,
        "detection_method": method,
    }


@app.get("/api/books")
def list_books():
    """Returns every book saved in the database so far, with generation progress."""
    return db.get_all_books()


@app.delete("/api/books/{book_id}")
def delete_book(book_id: int):
    """Deletes a book, its chapters/chunks from the database, and its audio files from disk."""
    book_data = db.get_book_with_chapters(book_id)
    if not book_data:
        raise HTTPException(status_code=404, detail="Book not found.")

    db.delete_book(book_id)

    book_audio_dir = os.path.join(AUDIO_DIR, f"book_{book_id}")
    if os.path.exists(book_audio_dir):
        shutil.rmtree(book_audio_dir)

    return {"message": "Book deleted."}


@app.get("/api/books/{book_id}")
def get_book(book_id: int):
    """Returns one saved book's full details, including all of its chapters."""
    result = db.get_book_with_chapters(book_id)
    if not result:
        raise HTTPException(status_code=404, detail="Book not found.")
    return result


@app.post("/api/books/{book_id}/chapters/{chapter_index}/approve")
def approve_chapter(book_id: int, chapter_index: int):
    """Marks a chapter as reviewed and approved by the user."""
    db.set_chapter_approved(book_id, chapter_index, True)
    return {"message": "Chapter approved."}


@app.post("/api/books/{book_id}/chapters/{chapter_index}/unapprove")
def unapprove_chapter(book_id: int, chapter_index: int):
    """Reverts a chapter's approval, in case you want to review it again."""
    db.set_chapter_approved(book_id, chapter_index, False)
    return {"message": "Chapter approval removed."}


@app.post("/api/books/{book_id}/chapters/{chapter_index}/chunk")
def create_chunks(book_id: int, chapter_index: int):
    """Splits one chapter's text into TTS-sized chunks and saves them."""
    book_data = db.get_book_with_chapters(book_id)
    if not book_data:
        raise HTTPException(status_code=404, detail="Book not found.")

    chapter = next(
        (c for c in book_data["chapters"] if c["chapter_index"] == chapter_index), None
    )
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found.")

    chunks = chunk_text(chapter["content"])
    db.save_chunks(book_id, chapter_index, chunks)

    return {"message": f"Created {len(chunks)} chunk(s).", "chunk_count": len(chunks)}


@app.get("/api/books/{book_id}/chapters/{chapter_index}/chunks")
def list_chunks(book_id: int, chapter_index: int):
    """Returns the saved chunks for one chapter, in order."""
    return db.get_chunks(book_id, chapter_index)


@app.get("/api/voices")
async def get_voices():
    """Returns the list of available English TTS voices."""
    return await tts.list_english_voices()


@app.post("/api/books/{book_id}/chapters/{chapter_index}/chunks/{chunk_index}/generate-audio")
async def generate_chunk_audio(book_id: int, chapter_index: int, chunk_index: int, voice: str = "en-US-AriaNeural"):
    """Converts one chunk's text into an MP3 file using the chosen voice."""
    chunks = db.get_chunks(book_id, chapter_index)
    chunk = next((c for c in chunks if c["chunk_index"] == chunk_index), None)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found. Try splitting into chunks first.")

    chapter_dir = os.path.join(AUDIO_DIR, f"book_{book_id}", f"chapter_{chapter_index}")
    os.makedirs(chapter_dir, exist_ok=True)
    output_path = os.path.join(chapter_dir, f"chunk_{chunk_index:03d}.mp3")

    db.set_chunk_status(book_id, chapter_index, chunk_index, status="processing")
    try:
        await tts.generate_speech("edge", chunk["text"], voice, output_path)
    except Exception as e:
        db.set_chunk_status(book_id, chapter_index, chunk_index, status="failed", error_message=str(e))
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {e}")

    audio_url = f"/audio/book_{book_id}/chapter_{chapter_index}/chunk_{chunk_index:03d}.mp3"
    db.set_chunk_status(book_id, chapter_index, chunk_index, status="completed", audio_path=audio_url, voice=voice, error_message=None)
    return {"message": "Audio generated.", "audio_url": audio_url}


cancel_flags = {}


async def process_chapter_chunks(book_id: int, chapter_index: int, voice: str, provider: str = "edge"):
    """
    Plain background worker: generates audio for every chunk in a chapter that
    isn't already completed. This function doesn't know anything about FastAPI
    or HTTP requests — it just does the work — which means later, swapping
    FastAPI's BackgroundTasks for a real job queue (Celery/RQ) only means
    changing how this function gets *called*, not what it does.
    """
    key = (book_id, chapter_index)
    chunks = db.get_chunks(book_id, chapter_index)
    chapter_dir = os.path.join(AUDIO_DIR, f"book_{book_id}", f"chapter_{chapter_index}")
    os.makedirs(chapter_dir, exist_ok=True)

    for chunk in chunks:
        if chunk["status"] == "completed":
            continue  # already done — lets us resume without redoing finished work

        chunk_index = chunk["chunk_index"]

        if cancel_flags.get(key):
            # A stop was requested — mark this and every remaining chunk as
            # cancelled instead of generating it, so progress tracking knows
            # there's no more work in flight.
            db.set_chunk_status(book_id, chapter_index, chunk_index, status="cancelled")
            continue

        db.set_chunk_status(book_id, chapter_index, chunk_index, status="processing")

        output_path = os.path.join(chapter_dir, f"chunk_{chunk_index:03d}.mp3")
        try:
            await tts.generate_speech(provider, chunk["text"], voice, output_path)
            audio_url = f"/audio/book_{book_id}/chapter_{chapter_index}/chunk_{chunk_index:03d}.mp3"
            db.set_chunk_status(book_id, chapter_index, chunk_index, status="completed", audio_path=audio_url, voice=voice, error_message=None)
        except Exception as e:
            db.set_chunk_status(book_id, chapter_index, chunk_index, status="failed", error_message=str(e))

    cancel_flags.pop(key, None)


@app.post("/api/books/{book_id}/chapters/{chapter_index}/generate-chapter-audio")
def generate_chapter_audio(book_id: int, chapter_index: int, background_tasks: BackgroundTasks, voice: str = "en-US-AriaNeural"):
    """Starts generating audio for every chunk in a chapter, in the background."""
    chunks = db.get_chunks(book_id, chapter_index)
    if not chunks:
        raise HTTPException(status_code=404, detail="No chunks found. Split the chapter into chunks first.")

    cancel_flags[(book_id, chapter_index)] = False  # clear any earlier stop request

    for chunk in chunks:
        if chunk["status"] != "completed":
            db.set_chunk_status(book_id, chapter_index, chunk["chunk_index"], status="pending")

    background_tasks.add_task(process_chapter_chunks, book_id, chapter_index, voice)
    return {"message": f"Started generating audio for {len(chunks)} chunk(s) in the background."}


@app.post("/api/books/{book_id}/chapters/{chapter_index}/cancel-generation")
def cancel_generation(book_id: int, chapter_index: int):
    """Requests that an in-progress chapter generation stop after the current chunk."""
    cancel_flags[(book_id, chapter_index)] = True
    return {"message": "Stopping — the current chunk will finish, then generation will stop."}


@app.get("/api/books/{book_id}/chapters/{chapter_index}/generation-status")
def get_generation_status(book_id: int, chapter_index: int):
    """Returns the current status of every chunk in a chapter, for progress polling."""
    chunks = db.get_chunks(book_id, chapter_index)
    return [
        {
            "chunk_index": c["chunk_index"],
            "status": c["status"],
            "audio_path": c["audio_path"],
            "error_message": c["error_message"],
        }
        for c in chunks
    ]


@app.post("/api/books/{book_id}/progress")
def save_progress(book_id: int, chapter_index: int, chunk_index: int, position_seconds: float):
    """Saves the current listening position so it can be resumed later."""
    db.save_playback_progress(book_id, chapter_index, chunk_index, position_seconds)
    return {"message": "Progress saved."}


@app.get("/api/books/{book_id}/progress")
def get_progress(book_id: int):
    """Returns the last saved listening position for a book, if any."""
    progress = db.get_playback_progress(book_id)
    if not progress:
        return {"chapter_index": None, "chunk_index": None, "position_seconds": 0}
    return progress


@app.post("/api/books/{book_id}/chapters/{chapter_index}/reset-from/{chunk_index}")
def reset_from_chunk(book_id: int, chapter_index: int, chunk_index: int):
    """Marks a chunk and every chunk after it for regeneration (e.g. to switch voice)."""
    db.reset_chunks_from(book_id, chapter_index, chunk_index)
    return {"message": f"Chunks from {chunk_index + 1} onward marked for regeneration."}


app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio_files")
app.mount("/", StaticFiles(directory="static", html=True), name="static")