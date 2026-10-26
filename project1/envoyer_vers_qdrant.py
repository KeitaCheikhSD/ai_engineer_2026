"""Envoie les chunks (texte + vecteur + métadonnées) dans une collection Qdrant.

Avant de lancer, définis deux variables d'environnement (jamais la clé dans le code) :
    QDRANT_URL       l'adresse de ton cluster
    QDRANT_API_KEY   ta clé d'API

    Windows (PowerShell) :  $env:QDRANT_API_KEY = "ta_cle"
    Mac / Linux          :  export QDRANT_API_KEY="ta_cle"

Lancer :  python envoyer_vers_qdrant.py
"""

import json
import os
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# ----------------------------------------------------------------------------
# Réglages
# ----------------------------------------------------------------------------
FICHIER_CHUNKS = Path("chunks.json")
COLLECTION = "items"
NOM_MODELE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PREFIXE_DOCUMENT = ""      # "passage: " si tu passes à un modèle e5
PREFIXE_QUESTION = ""      # "query: "   si tu passes à un modèle e5
SUPPRIMER_OBSOLETES = True # retire de Qdrant les chunks qui n'existent plus dans le fichier


# ----------------------------------------------------------------------------
# Connexion
# ----------------------------------------------------------------------------
def connecter():
    return QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )


# ----------------------------------------------------------------------------
# Collection
# ----------------------------------------------------------------------------
def preparer_collection(client, dimension):
    """Crée la collection si elle n'existe pas. Vérifie la taille des vecteurs sinon."""
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
        return
    taille = client.get_collection(COLLECTION).config.params.vectors.size
    if taille != dimension:
        raise SystemExit(
            f"La collection '{COLLECTION}' attend des vecteurs de taille {taille}, "
            f"mais ton modèle en produit {dimension}. Change de collection ou supprime-la."
        )


# ----------------------------------------------------------------------------
# Points (un point = un id + un vecteur + un payload)
# ----------------------------------------------------------------------------
def id_qdrant(chunk_id):
    """Qdrant veut un entier ou un UUID : on dérive un UUID stable de l'id du chunk."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(chunk_id)))


def construire_points(chunks, vecteurs):
    return [
        PointStruct(
            id=id_qdrant(c["id"]),
            vector=v.tolist(),
            payload={
                **c.get("metadata", {}),     # d'abord les métadonnées...
                "chunk_id": c["id"],         # ...puis nos clés, pour qu'elles gagnent
                "text": c["text"],
                "source": c.get("source"),
            },
        )
        for c, v in zip(chunks, vecteurs)
    ]


def supprimer_obsoletes(client, ids_voulus):
    """Supprime les points de Qdrant dont l'id n'est plus dans le fichier de chunks."""
    a_supprimer, offset = [], None
    while True:
        lot, offset = client.scroll(
            COLLECTION, limit=256, offset=offset, with_payload=False, with_vectors=False
        )
        a_supprimer += [p.id for p in lot if str(p.id) not in ids_voulus]
        if offset is None:
            break
    if a_supprimer:
        client.delete(COLLECTION, points_selector=a_supprimer)
    return len(a_supprimer)


# ----------------------------------------------------------------------------
# Programme principal
# ----------------------------------------------------------------------------
def main(client=None, modele=None):
    chunks = json.loads(FICHIER_CHUNKS.read_text(encoding="utf-8"))
    if isinstance(chunks, dict):
        chunks = [chunks]

    if modele is None:
        from sentence_transformers import SentenceTransformer
        modele = SentenceTransformer(NOM_MODELE)
    client = client or connecter()

    textes = [PREFIXE_DOCUMENT + c["text"] for c in chunks]

    # Contrôle : les textes dépassent-ils la limite du modèle ? (ils seraient coupés)
    limite = modele.max_seq_length
    trop_longs = sum(len(modele.tokenizer(t)["input_ids"]) > limite for t in textes)
    print(f"Limite du modèle : {limite} tokens, {trop_longs} chunk(s) la dépassent")

    vecteurs = modele.encode(textes, show_progress_bar=True)
    preparer_collection(client, modele.get_sentence_embedding_dimension())

    points = construire_points(chunks, vecteurs)
    client.upload_points(collection_name=COLLECTION, points=points, batch_size=64)
    print(f"{len(points)} chunks envoyés dans la collection '{COLLECTION}'")

    if SUPPRIMER_OBSOLETES:
        n = supprimer_obsoletes(client, {p.id for p in points})
        print(f"{n} point(s) obsolète(s) supprimé(s)")


if __name__ == "__main__":
    main()
