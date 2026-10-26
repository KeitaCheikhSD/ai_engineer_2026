"""Pose des questions à ta collection Qdrant.

Lancer :  python chercher_qdrant.py      (après envoyer_vers_qdrant.py)
"""

from envoyer_vers_qdrant import COLLECTION, NOM_MODELE, PREFIXE_QUESTION, connecter

NB_RESULTATS = 5


def chercher(client, modele, question, k=NB_RESULTATS):
    """Renvoie les k points les plus proches de la question."""
    vecteur = modele.encode(PREFIXE_QUESTION + question).tolist()
    reponse = client.query_points(
        collection_name=COLLECTION,
        query=vecteur,
        limit=k,
        with_payload=True,
    )
    return reponse.points


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    modele = SentenceTransformer(NOM_MODELE)   # LE MÊME modèle que pour l'envoi
    client = connecter()

    while True:
        question = input("\nQuestion (Entrée pour quitter) : ").strip()
        if not question:
            break
        for p in chercher(client, modele, question):
            apercu = p.payload["text"][:200].replace("\n", " ")
            print(f"\n[{p.score:.3f}] {p.payload['source']} - {p.payload.get('title', '')}")
            print(f"   {apercu}...")
