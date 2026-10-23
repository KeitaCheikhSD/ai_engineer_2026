"""Phase 0 (version simple) : transformer des fichiers bruts en chunks propres.

Lancer :  python prepare_corpus_simple.py
Résultat : un fichier chunks_simple.json
"""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------------------------------------------------------
# Réglages (on pourra les modifier pour tester)
# ----------------------------------------------------------------------------
DOSSIER_CORPUS = Path("corpus")
FICHIER_SORTIE = Path("chunks_simple.json")
TAILLE_MAX = 800        # taille maximale d'un chunk, en caractères
CHEVAUCHEMENT = 100     # caractères répétés entre deux chunks voisins
TAILLE_MIN = 30         # on jette les chunks plus courts que ça


# ----------------------------------------------------------------------------
# Étape 1 : CHARGER les fichiers
# Chaque "document" est un dictionnaire : source, titre, texte, métadonnées.
# ----------------------------------------------------------------------------
def charger_qcm(chemin):
    """Lit le JSON du QCM. Une question avec ses réponses = un document."""
    donnees = json.loads(chemin.read_text(encoding="utf-8"))
    documents = []
    for bloc in donnees["data"]:                      # un bloc = un thème
        for q in bloc["questions"]:
            reponses = "\n".join("- " + r for r in q["suggested_answers"])
            texte = (
                f"Thème : {bloc['theme']}\n"
                f"Question : {q['question']}\n"
                f"Réponses :\n{reponses}"
            )
            documents.append({
                "source": chemin.name,
                "titre": q["question"],
                "texte": texte,
                "meta": {"theme": bloc["theme"], "question_id": q["id"]},
            })
    return documents


def charger_textes(dossier):
    """Lit tous les fichiers .md et .txt d'un dossier. Un fichier = un document."""
    documents = []
    chemins = sorted(dossier.glob("*.md")) + sorted(dossier.glob("*.txt"))
    for chemin in chemins:
        documents.append({
            "source": chemin.name,
            "titre": chemin.stem,
            "texte": chemin.read_text(encoding="utf-8"),
            "meta": {},
        })
    return documents


# ----------------------------------------------------------------------------
# Étape 2 : NETTOYER le texte
# ----------------------------------------------------------------------------
def nettoyer(texte):
    texte = unicodedata.normalize("NFKC", texte)   # unifie les caractères Unicode
    texte = texte.replace("’", "'")                # apostrophe courbe -> droite
    texte = re.sub(r"<[^>]+>", " ", texte)         # supprime les balises HTML
    texte = re.sub(r"[ \t]+", " ", texte)          # plusieurs espaces -> un seul
    texte = re.sub(r" *\n *", "\n", texte)         # espaces autour des retours ligne
    texte = re.sub(r"\n{3,}", "\n\n", texte)       # trop de lignes vides -> une seule
    return texte.strip()


# ----------------------------------------------------------------------------
# Étape 3 : DÉCOUPER en chunks
# 1) on coupe d'abord par sections (chaque titre "## ..." commence une section)
# 2) si une section est encore trop longue, on la redécoupe
# ----------------------------------------------------------------------------
decoupeur = RecursiveCharacterTextSplitter(
    chunk_size=TAILLE_MAX,
    chunk_overlap=CHEVAUCHEMENT,
    separators=["\n\n", "\n", ". ", " ", ""],   # du plus gros au plus petit
    keep_separator="end",                       # le point reste à la fin de sa phrase
)


def decouper(texte):
    chunks = []
    sections = re.split(r"\n(?=#{1,6} )", texte)    # coupe juste avant chaque titre
    for section in sections:
        section = section.strip()
        if len(section) <= TAILLE_MAX:
            chunks.append(section)                  # section assez courte : on garde
        else:
            # Section trop longue : on met le titre de côté, on découpe le reste,
            # puis on recolle le titre devant chaque morceau.
            premiere_ligne, _, corps = section.partition("\n")
            if premiere_ligne.startswith("#"):
                titre, corps = premiere_ligne, corps
            else:
                titre, corps = "", section
            for morceau in decoupeur.split_text(corps):
                chunks.append((titre + "\n\n" + morceau).strip())
    return chunks


# ----------------------------------------------------------------------------
# Étape 4 : FABRIQUER les chunks (id + métadonnées)
# ----------------------------------------------------------------------------
def fabriquer_id(source, position, texte):
    """Même entrée -> même id, à chaque exécution (pas de hasard)."""
    brut = f"{source}|{position}|{texte}"
    return hashlib.sha256(brut.encode("utf-8")).hexdigest()[:16]


def fabriquer_chunks(document):
    resultats = []
    textes = decouper(document["texte"])
    for position, texte in enumerate(textes):
        if len(texte) < TAILLE_MIN:                 # trop court : inutile
            continue
        resultats.append({
            "id": fabriquer_id(document["source"], position, texte),
            "text": texte,
            "source": document["source"],
            "metadata": {
                "title": document["titre"],
                "date": None,
                "chunk_index": position,
                **document["meta"],
            },
        })
    return resultats


# ----------------------------------------------------------------------------
# Étape 5 : SAUVEGARDER en JSON
# ----------------------------------------------------------------------------
def sauvegarder(chunks, chemin):
    sans_doublon = {c["id"]: c for c in chunks}     # même id = une seule entrée
    liste = list(sans_doublon.values())
    chemin.write_text(json.dumps(liste, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(liste)


# ----------------------------------------------------------------------------
# Programme principal : on enchaîne les étapes
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    documents = []
    documents += charger_qcm(DOSSIER_CORPUS / "data-gouv-qcm-civique-resident.json")
    documents += charger_textes(DOSSIER_CORPUS)

    tous_les_chunks = []
    for doc in documents:
        doc["texte"] = nettoyer(doc["texte"])
        doc["titre"] = nettoyer(doc["titre"])
        tous_les_chunks += fabriquer_chunks(doc)

    n = sauvegarder(tous_les_chunks, FICHIER_SORTIE)
    print(f"{len(documents)} documents lus -> {n} chunks écrits dans {FICHIER_SORTIE}")