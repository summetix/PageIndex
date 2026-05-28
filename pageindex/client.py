import os
import uuid
import json
import asyncio
import concurrent.futures
from pathlib import Path
import re
from datetime import datetime, timezone, timedelta
import PyPDF2
import pymupdf

from .page_index import page_index
from .page_index_md import md_to_tree
from .retrieve import (
    get_children,
    get_document,
    get_document_structure,
    get_page_content,
)
from .utils import ConfigLoader, remove_fields

META_INDEX = "_meta.json"

PDF_DATE_RE = re.compile(
    r"^D:"
    r"(?P<year>\d{4})"
    r"(?P<month>\d{2})?"
    r"(?P<day>\d{2})?"
    r"(?P<hour>\d{2})?"
    r"(?P<minute>\d{2})?"
    r"(?P<second>\d{2})?"
    r"(?P<tz>Z|[+-]\d{2}'?\d{2}'?)?"
)

def _normalize_retrieve_model(model: str) -> str:
    """Preserve supported Agents SDK prefixes and route other provider paths via LiteLLM."""
    passthrough_prefixes = ("litellm/", "openai/")
    if not model or "/" not in model:
        return model
    if model.startswith(passthrough_prefixes):
        return model
    return f"litellm/{model}"


class PageIndexClient:
    """
    A client for indexing and retrieving document content.
    Flow: index() -> get_document() / get_document_structure() / get_page_content()

    For agent-based QA, see examples/agentic_vectorless_rag_demo.py.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        retrieve_model: str = None,
        workspace: str = None,
    ):
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif not os.getenv("OPENAI_API_KEY") and os.getenv("CHATGPT_API_KEY"):
            os.environ["OPENAI_API_KEY"] = os.getenv("CHATGPT_API_KEY")
        self.workspace = Path(workspace).expanduser() if workspace else None
        overrides = {}
        if model:
            overrides["model"] = model
        if retrieve_model:
            overrides["retrieve_model"] = retrieve_model
        opt = ConfigLoader().load(overrides or None)
        self.model = opt.model
        self.retrieve_model = _normalize_retrieve_model(
            opt.retrieve_model or self.model
        )
        if self.workspace:
            self.workspace.mkdir(parents=True, exist_ok=True)
        self.documents = {}
        if self.workspace:
            self._load_workspace()

    def doc_id_from_filename(self, path_or_filename: str, pageindex_namespace: str) -> str:
        filename = Path(path_or_filename).name.strip().lower()
        return str(uuid.uuid5(pageindex_namespace, filename))


    def parse_pdf_date(self, value: str | None) -> datetime | None:
        """
        Parses PDF dates like:
          D:20240131153000+01'00'
          D:20240131153000Z
          D:20240131153000
        Returns UTC datetime.
        """
        if not value:
            return None

        value = value.strip()
        match = PDF_DATE_RE.match(value)
        if not match:
            return None

        parts = match.groupdict()

        try:
            year = int(parts["year"])
            month = int(parts["month"] or 1)
            day = int(parts["day"] or 1)
            hour = int(parts["hour"] or 0)
            minute = int(parts["minute"] or 0)
            second = int(parts["second"] or 0)
        except ValueError:
            return None

        tz_value = parts.get("tz")

        if tz_value == "Z":
            tzinfo = timezone.utc
        elif tz_value and (tz_value.startswith("+") or tz_value.startswith("-")):
            sign = 1 if tz_value[0] == "+" else -1
            digits = tz_value[1:].replace("'", "")

            if len(digits) != 4:
                tzinfo = timezone.utc
            else:
                offset_hours = int(digits[:2])
                offset_minutes = int(digits[2:])
                tzinfo = timezone(
                    sign * timedelta(hours=offset_hours, minutes=offset_minutes)
                )
        else:
            # If PDF metadata has no timezone, choose one consistent convention.
            # UTC is usually safest for indexing.
            tzinfo = timezone.utc

        try:
            return datetime(
                year,
                month,
                day,
                hour,
                minute,
                second,
                tzinfo=tzinfo,
            ).astimezone(timezone.utc)
        except ValueError:
            return None

    def format_timestamp_utc(self, dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt.astimezone(timezone.utc).strftime(os.getenv("FORMAT_SOURCE_TIMESTAMP", "%Y-%m-%dT%H:%M:%SZ"))

    def infer_timestamp_from_file(self, file_path: str) -> tuple[str, str]:
        """
        Infer a stable timestamp from the file itself.

        Priority:
          1. PDF creationDate metadata
          2. PDF modDate metadata
          3. filesystem mtime fallback

        Returns:
          (timestamp_iso_utc, source)
        """
        file_path = os.path.abspath(os.path.expanduser(file_path))
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            try:
                doc = pymupdf.open(file_path)
                try:
                    metadata = doc.metadata or {}

                    creation_date = self.parse_pdf_date(metadata.get("creationDate"))
                    if creation_date is not None:
                        return creation_date.isoformat(), "pdf.creationDate"

                    modification_date = self.parse_pdf_date(metadata.get("modDate"))
                    if modification_date is not None:
                        return modification_date.isoformat(), "pdf.modDate"

                finally:
                    doc.close()

            except Exception:
                pass

        mtime = os.path.getmtime(file_path)
        fallback_dt = datetime.fromtimestamp(mtime, timezone.utc)
        return self.format_timestamp_utc(fallback_dt), "filesystem.mtime"

    def index(self, file_path: str, pageindex_namespace: str, mode: str = "auto") -> str:
        """Index a document. Returns a document_id."""
        # Persist a canonical absolute path so workspace reloads do not
        # reinterpret caller-relative paths against the workspace directory.
        file_path = os.path.abspath(os.path.expanduser(file_path))
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        document_timestamp, timestamp_source = self.infer_timestamp_from_file(file_path)

        ext = os.path.splitext(file_path)[1].lower()
        doc_id = self.doc_id_from_filename(os.path.basename(file_path), pageindex_namespace)

        is_pdf = ext == ".pdf"
        is_md = ext in [".md", ".markdown"]

        if mode == "pdf" or (mode == "auto" and is_pdf):
            print(f"Indexing PDF: {file_path}")
            result = page_index(
                doc=file_path,
                model=self.model,
                if_add_node_summary="yes",
                if_add_node_text="yes",
                if_add_node_id="yes",
                if_add_doc_description="yes",
            )
            # Extract per-page text so queries don't need the original PDF
            pages = []
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(pdf_reader.pages, 1):
                    pages.append({"page": i, "content": page.extract_text() or ""})

            self.documents[doc_id] = {
                "id": doc_id,
                "type": "pdf",
                "path": file_path,
                "timestamp": document_timestamp,
                "doc_name": result.get("doc_name", ""),
                "doc_description": result.get("doc_description", ""),
                "page_count": len(pages),
                "structure": result["structure"],
                "pages": pages,
            }

        elif mode == "md" or (mode == "auto" and is_md):
            print(f"Indexing Markdown: {file_path}")
            coro = md_to_tree(
                md_path=file_path,
                if_thinning=False,
                if_add_node_summary="yes",
                summary_token_threshold=200,
                model=self.model,
                if_add_doc_description="yes",
                if_add_node_text="yes",
                if_add_node_id="yes",
            )
            try:
                asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(asyncio.run, coro).result()
            except RuntimeError:
                result = asyncio.run(coro)
            self.documents[doc_id] = {
                "id": doc_id,
                "type": "md",
                "path": file_path,
                "timestamp": document_timestamp,
                "doc_name": result.get("doc_name", ""),
                "doc_description": result.get("doc_description", ""),
                "line_count": result.get("line_count", 0),
                "structure": result["structure"],
            }
        else:
            raise ValueError(f"Unsupported file format for: {file_path}")

        print(f"Indexing complete. Document ID: {doc_id}")
        if self.workspace:
            self._save_doc(doc_id)
        return doc_id

    @staticmethod
    def _make_meta_entry(doc: dict) -> dict:
        """Build a lightweight meta entry from a document dict."""
        entry = {
            "type": doc.get("type", ""),
            "doc_name": doc.get("doc_name", ""),
            "doc_description": doc.get("doc_description", ""),
            "path": doc.get("path", ""),
        }
        if doc.get("type") == "pdf":
            entry["page_count"] = doc.get("page_count")
        elif doc.get("type") == "md":
            entry["line_count"] = doc.get("line_count")
        return entry

    @staticmethod
    def _read_json(path) -> dict | None:
        """Read a JSON file, returning None on any error."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: corrupt {Path(path).name}: {e}")
            return None

    def _save_doc(self, doc_id: str):
        doc = self.documents[doc_id].copy()
        # Strip text from structure nodes — redundant with pages (PDF only)
        if doc.get("structure") and doc.get("type") == "pdf":
            doc["structure"] = remove_fields(doc["structure"], fields=["text"])
        path = self.workspace / f"{doc_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        self._save_meta(doc_id, self._make_meta_entry(doc))
        # Drop heavy fields; will lazy-load on demand
        self.documents[doc_id].pop("structure", None)
        self.documents[doc_id].pop("pages", None)

    def _rebuild_meta(self) -> dict:
        """Scan individual doc JSON files and return a meta dict."""
        meta = {}
        for path in self.workspace.glob("*.json"):
            if path.name == META_INDEX:
                continue
            doc = self._read_json(path)
            if doc and isinstance(doc, dict):
                meta[path.stem] = self._make_meta_entry(doc)
        return meta

    def _read_meta(self) -> dict | None:
        """Read and validate _meta.json, returning None on any corruption."""
        meta = self._read_json(self.workspace / META_INDEX)
        if meta is not None and not isinstance(meta, dict):
            print(f"Warning: {META_INDEX} is not a JSON object, ignoring")
            return None
        return meta

    def _save_meta(self, doc_id: str, entry: dict):
        meta = self._read_meta() or self._rebuild_meta()
        meta[doc_id] = entry
        meta_path = self.workspace / META_INDEX
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _load_workspace(self):
        meta = self._read_meta()
        if meta is None:
            meta = self._rebuild_meta()
            if meta:
                print(f"Loaded {len(meta)} document(s) from workspace (legacy mode).")
        for doc_id, entry in meta.items():
            doc = dict(entry, id=doc_id)
            if doc.get("path") and not os.path.isabs(doc["path"]):
                doc["path"] = str((self.workspace / doc["path"]).resolve())
            self.documents[doc_id] = doc

    def _ensure_doc_loaded(self, doc_id: str):
        """Load full document JSON on demand (structure, pages, etc.)."""
        doc = self.documents.get(doc_id)
        if not doc or doc.get("structure") is not None:
            return
        full = self._read_json(self.workspace / f"{doc_id}.json")
        if not full:
            return
        doc["structure"] = full.get("structure", [])
        if full.get("pages"):
            doc["pages"] = full["pages"]

    def get_document(self, doc_id: str) -> str:
        """Return document metadata JSON."""
        return get_document(self.documents, doc_id)

    def get_document_structure(
        self,
        doc_id: str,
        parts: int = 1,
        depth: int | None = None,
        token_limit: int | None = None,
        token_model: str | None = None,
    ) -> str:
        """Return tree structure JSON (with optional depth truncation and top-level pagination)."""
        if self.workspace:
            self._ensure_doc_loaded(doc_id)
        return get_document_structure(
            self.documents,
            doc_id,
            parts=parts,
            depth=depth,
            token_limit=token_limit,
            token_model=token_model,
        )

    def get_children(
        self,
        doc_id: str,
        node_id: str = "",
        node_path: str = "",
        depth: int = 1,
    ) -> str:
        """Return children for one node, addressed by node_id or node_path."""
        if self.workspace:
            self._ensure_doc_loaded(doc_id)
        return get_children(
            self.documents,
            doc_id,
            node_id=node_id,
            node_path=node_path,
            depth=depth,
        )

    def get_page_content(self, doc_id: str, pages: str) -> str:
        """Return page content for the given pages string (e.g. '5-7', '3,8', '12')."""
        if self.workspace:
            self._ensure_doc_loaded(doc_id)
        return get_page_content(self.documents, doc_id, pages)
