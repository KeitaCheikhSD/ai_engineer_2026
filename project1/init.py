"""Phase 0 : prépare un corpus de chunks propres à partir de sources brutes.

Flux :  Loader -> Document -> TextCleaner -> Chunker -> Chunk -> ChunkStore (chunks.json)

Usage :
    python prepare_corpus.py
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator


# --------------------------------------------------------------------------- #
# 1. Structures de données                                                    #
# --------------------------------------------------------------------------- #

@dataclass
class Document:
    """Une unité brute lue depuis une source, avant nettoyage et découpage."""
    source: str                      # chemin ou URL d'origine
    title: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """Unité finale qui sera transformée en vecteur en Phase 1."""
    id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def make_id(source: str, position: str, text: str) -> str:
            """ID déterministe : mêmes entrées => même ID, à chaque exécution.

            - `position` distingue deux chunks identiques situés à des endroits différents.
            - `text` fait changer l'ID si le contenu source a été modifié.
            """
            raw = f"{source}|{position}|{text}".encode("utf-8")
            return hashlib.sha256(raw).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# 2. Nettoyage                                                                #
# --------------------------------------------------------------------------- #

class TextCleaner:
    """Normalise un texte : Unicode, apostrophes, espaces, balises HTML."""

    _HTML = re.compile(r"<[^>]+>")
    _SPACES = re.compile(r"[ \t\u00a0]+")
    _BLANK_LINES = re.compile(r"\n{3,}")

    def __init__(self, strip_html: bool = True):
        self.strip_html = strip_html

    def clean(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)   # unifie les formes Unicode
        text = text.replace("\u2019", "'")           # ’ -> '
        if self.strip_html:
            text = self._HTML.sub(" ", text)
        text = self._SPACES.sub(" ", text)           # espaces multiples et insécables
        text = re.sub(r" *\n *", "\n", text)         # espaces autour des retours ligne
        text = self._BLANK_LINES.sub("\n\n", text)
        return text.strip()


# --------------------------------------------------------------------------- #
# 3. Chargement (une classe par type de source, même interface)               #
# --------------------------------------------------------------------------- #

class BaseLoader(ABC):
    """Interface commune : tout loader produit des Document."""

    @abstractmethod
    def load(self) -> Iterator[Document]:
        ...


class QcmJsonLoader(BaseLoader):
    """Lit le JSON du QCM civique : 1 question = 1 Document."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> Iterator[Document]:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        meta = data.get("metadata", {})
        for block in data["data"]:
            for q in block["questions"]:
                yield Document(
                    source=self.path.name,
                    title=q["question"],
                    text="\n".join(q["suggested_answers"]),
                    metadata={
                        "theme": block["theme"],
                        "question_id": q["id"],
                        "answers": q["suggested_answers"],
                        "licence": meta.get("licence"),
                        "answers_author": meta.get("auteur_reponses_suggerees"),
                        "date": None,
                    },
                )


class TextFileLoader(BaseLoader):
    """Lit tous les .md / .txt d'un dossier (batch) : 1 fichier = 1 Document."""

    def __init__(self, folder: Path, patterns: tuple[str, ...] = ("*.md", "*.txt")):
        self.folder = Path(folder)
        self.patterns = patterns

    def load(self) -> Iterator[Document]:
        files = sorted(p for pat in self.patterns for p in self.folder.glob(pat))
        for path in files:
            text = path.read_text(encoding="utf-8")
            first = text.lstrip().splitlines()[0] if text.strip() else path.stem
            title = first.lstrip("# ").strip() or path.stem
            yield Document(source=path.name, title=title, text=text,
                           metadata={"date": None})


# class PdfLoader(BaseLoader): ...   # à écrire avec pypdf / pdfplumber
# class WebLoader(BaseLoader): ...   # à écrire avec requests + beautifulsoup4


# --------------------------------------------------------------------------- #
# 4. Découpage                                                                #
# --------------------------------------------------------------------------- #

class BaseChunker(ABC):
    """Interface commune : un Document nettoyé devient une liste de Chunk."""

    @abstractmethod
    def split(self, doc: Document) -> list[Chunk]:
        ...


class QcmChunker(BaseChunker):
    """Un chunk par question, avec son thème et ses réponses.

    Une réponse seule ("Citoyens") n'a aucun sens sans sa question,
    donc on garde tout ensemble dans le même chunk.
    """

    def __init__(self, include_theme: bool = True):
        self.include_theme = include_theme

    def split(self, doc: Document) -> list[Chunk]:
        parts = []
        if self.include_theme:
            parts.append(f"Thème : {doc.metadata['theme']}")
        parts.append(f"Question : {doc.title}")
        parts.append("Réponses :")
        parts.extend(f"- {a}" for a in doc.metadata["answers"])
        text = "\n".join(parts)

        position = f"q{doc.metadata['question_id']}"
        return [Chunk(
            id=Chunk.make_id(doc.source, position, text),
            text=text,
            source=doc.source,
            metadata={
                "title": doc.title,
                "date": doc.metadata.get("date"),
                "theme": doc.metadata["theme"],
                "question_id": doc.metadata["question_id"],
                "licence": doc.metadata.get("licence"),
                "answers_author": doc.metadata.get("answers_author"),
            },
        )]


class SectionChunker(BaseChunker):
    """Découpe en deux niveaux : d'abord par sections Markdown, puis par taille.

    Niveau 1 : chaque titre (#, ##, ...) ouvre une nouvelle section, donc un
               nouveau chunk. Un titre n'est jamais séparé de son contenu.
    Niveau 2 : si une section dépasse `max_chars`, on la redécoupe avec
               RecursiveCharacterTextSplitter (langchain-text-splitters) :

        paragraphes  ->  lignes  ->  fins de phrase  ->  mots  ->  caractères

               avec `overlap` caractères répétés entre sous-chunks voisins.

    Pas de chevauchement entre deux sections différentes : chacune est
    autonome. Un titre sans contenu (ex. "# Titre" suivi de "## Sous-titre")
    est fusionné avec la section suivante.
    """

    _HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)

    SEPARATORS = [
        r"\n\n",               # paragraphes
        r"\n",                  # lignes
        r"(?<=[.!?])\s+",       # fin de phrase (le point reste avec la phrase)
        r" ",                   # mots
        "",                     # dernier recours : caractères
    ]

    _LEADING_HEADINGS = re.compile(r"\A(?:#{1,6}[ \t]+.+(?:\n|\Z)\s*)+")

    def __init__(self, max_chars: int = 800, overlap: int = 120):
        self.max_chars = max_chars
        self.overlap = overlap

    def _make_splitter(self, chunk_size: int):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=min(self.overlap, chunk_size // 4),
            separators=self.SEPARATORS,
            is_separator_regex=True,
            keep_separator="end",
            strip_whitespace=True,
            add_start_index=True,       # position du sous-chunk dans le texte découpé
        )

    def _sections(self, text: str) -> list[tuple[int, str, list[str]]]:
        """Renvoie [(position_de_début, texte_de_la_section, chemin_des_titres)]."""
        matches = list(self._HEADING.finditer(text))
        if not matches:
            return [(0, text, [])]
        sections = []
        if text[:matches[0].start()].strip():           # texte avant le 1er titre
            sections.append((0, text[:matches[0].start()], []))
        path: list[str] = []
        for i, m in enumerate(matches):
            level = len(m.group(1))
            path = path[:level - 1] + [m.group(2)]      # ex. ["Laïcité", "Le principe"]
            stop = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections.append((m.start(), text[m.start():stop], list(path)))
        return sections

    def split(self, doc: Document) -> list[Chunk]:
        raw: list[tuple[int, str, str]] = []            # (start, texte, chemin)
        pending: tuple[int, str] | None = None          # titre(s) sans contenu

        for start, sec_text, path in self._sections(doc.text):
            if pending:                                  # on colle le titre orphelin
                start, sec_text = pending[0], pending[1] + sec_text
                pending = None
            if not self._HEADING.sub("", sec_text).strip():   # section = titre seul
                pending = (start, sec_text)
                continue
            lead = len(sec_text) - len(sec_text.lstrip())
            body = sec_text.strip()
            section = " > ".join(path)
            if len(body) <= self.max_chars:
                raw.append((start + lead, body, section))
            else:
                # Section trop longue : le titre est collé au 1er sous-chunk,
                # on réserve donc sa place dans le budget de taille.
                m = self._LEADING_HEADINGS.match(body)
                heading = m.group(0).rstrip() if m else ""
                rest = body[len(m.group(0)):] if m else body
                budget = max(200, self.max_chars - len(heading) - 2) if heading else self.max_chars
                splitter = self._make_splitter(budget)
                offset = start + lead + (len(m.group(0)) if m else 0)
                for j, piece in enumerate(splitter.create_documents([rest])):
                    text = f"{heading}\n\n{piece.page_content}" if (j == 0 and heading) else piece.page_content
                    raw.append((offset + piece.metadata["start_index"], text, section))
        if pending:
            raw.append((pending[0], pending[1].strip(), ""))

        return [
            Chunk(
                id=Chunk.make_id(doc.source, f"c{i}", text),
                text=text,
                source=doc.source,
                metadata={
                    "title": doc.title,
                    "date": doc.metadata.get("date"),
                    "section": section,
                    "chunk_index": i,
                    "n_chunks": len(raw),
                    "start_char": start,
                },
            )
            for i, (start, text, section) in enumerate(raw)
        ]


# --------------------------------------------------------------------------- #
# 5. Sauvegarde                                                               #
# --------------------------------------------------------------------------- #

class ChunkStore:
    """Écrit et relit chunks.json. Dédoublonne par ID."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def save(self, chunks: list[Chunk]) -> int:
        unique = {c.id: c for c in chunks}           # même ID => une seule entrée
        ordered = list(unique.values())              # garde l'ordre de lecture des documents
        payload = [asdict(c) for c in ordered]
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return len(ordered)

    def load(self) -> list[Chunk]:
        return [Chunk(**d) for d in json.loads(self.path.read_text(encoding="utf-8"))]


# --------------------------------------------------------------------------- #
# 6. Orchestration                                                            #
# --------------------------------------------------------------------------- #

class CorpusPipeline:
    """Enchaîne, pour chaque source : loader -> nettoyage -> chunker -> filtre.

    Tous les chunks sont ensuite sauvegardés ensemble dans un seul chunks.json.
    """

    def __init__(self, sources: list[tuple[BaseLoader, BaseChunker]],
                 cleaner: TextCleaner, store: ChunkStore, min_chars: int = 30):
        self.sources = sources
        self.cleaner = cleaner
        self.store = store
        self.min_chars = min_chars

    def run(self) -> int:
        chunks: list[Chunk] = []
        skipped = 0
        for loader, chunker in self.sources:
            for doc in loader.load():
                doc.title = self.cleaner.clean(doc.title)
                doc.text = self.cleaner.clean(doc.text)
                if "answers" in doc.metadata:            # spécifique au QCM
                    doc.metadata["answers"] = [
                        self.cleaner.clean(a) for a in doc.metadata["answers"]
                    ]
                for chunk in chunker.split(doc):
                    if len(chunk.text) < self.min_chars:
                        skipped += 1
                        continue
                    chunks.append(chunk)
        n = self.store.save(chunks)
        print(f"{n} chunks écrits dans {self.store.path} ({skipped} ignorés car trop courts)")
        return n


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    base = Path(__file__).parent
    corpus = base / "corpus"
    pipeline = CorpusPipeline(
        sources=[
            (QcmJsonLoader(corpus / "data-gouv-qcm-civique-resident.json"),
             QcmChunker(include_theme=True)),
            (TextFileLoader(corpus),
             SectionChunker(max_chars=800, overlap=120)),
        ],
        cleaner=TextCleaner(),
        store=ChunkStore(base / "chunks.json"),
    )
    pipeline.run()