
"""
RAG : recherche dans Qdrant puis génération avec Qwen3.

Lancer :
    python chercher_qdrant.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from envoyer_vers_qdrant import (
    COLLECTION,
    NOM_MODELE,
    PREFIXE_QUESTION,
    connecter,
)

# ---------------------------
# Configuration
# ---------------------------

MODELE_IA = "Qwen/Qwen3-0.6B"
NB_RESULTATS = 5
MAX_NOUVEAUX_TOKENS = 512

# Ajuste ce seuil selon les scores observés dans ta collection.
# Mets None pour ne pas filtrer les résultats.
SCORE_MINIMUM = None

CONTEXTE_SYSTEME = """
Tu es un assistant pédagogique qui aide l'utilisateur à comprendre
les réponses aux questions de citoyenneté française.

Ton objectif est d'expliquer le pourquoi et le comment, pas seulement
de donner une réponse courte.

Consignes :
- Réponds en français, avec un langage simple et accessible.
- Commence par répondre directement à la question.
- Explique ensuite pourquoi cette réponse est correcte.
- Utilise des exemples concrets si cela facilite la compréhension.
- Appuie-toi sur les documents fournis comme source de contexte.
- Ne présente pas une information absente des documents comme
  si elle était confirmée par ceux-ci.
- Si le contexte ne suffit pas, indique-le clairement.
- Si les documents se contredisent, signale cette limite.
- Ne recopie pas inutilement tous les documents.
- Distingue la réponse attendue dans un QCM de son explication.
"""


# ---------------------------
# Chargement des modèles
# ---------------------------

def charger_qwen():
    print("Chargement du modèle Qwen...")

    tokenizer = AutoTokenizer.from_pretrained(MODELE_IA)

    model = AutoModelForCausalLM.from_pretrained(
        MODELE_IA,
        torch_dtype="auto",
        device_map="auto",
    )

    model.eval()
    return tokenizer, model


# ---------------------------
# Recherche dans Qdrant
# ---------------------------

def chercher(client, embedding_model, question, k=NB_RESULTATS):
    """Renvoie les k documents les plus proches de la question."""

    vecteur = embedding_model.encode(
        PREFIXE_QUESTION + question
    ).tolist()

    reponse = client.query_points(
        collection_name=COLLECTION,
        query=vecteur,
        limit=k,
        with_payload=True,
    )

    return reponse.points


# ---------------------------
# Construction du contexte
# ---------------------------

def construire_contexte(points):
    """Transforme les résultats Qdrant en contexte lisible par Qwen."""

    documents = []

    for i, point in enumerate(points, start=1):
        if SCORE_MINIMUM is not None and point.score < SCORE_MINIMUM:
            continue

        payload = point.payload or {}

        texte = payload.get("text", "")
        titre = payload.get("title", "")
        source = payload.get("source", "Source inconnue")

        if not texte and not titre:
            continue

        document = (
            f"Document {i}\n"
            f"Source : {source}\n"
            f"Titre : {titre}\n"
            f"Score de pertinence : {point.score:.3f}\n"
            f"Contenu : {texte or titre}"
        )

        documents.append(document)

    if not documents:
        return "Aucun document pertinent n'a été retrouvé."

    return "\n\n---\n\n".join(documents)


# ---------------------------
# Génération de la réponse IA
# ---------------------------

def generer_reponse(question, contexte, tokenizer, model):

    message_utilisateur = f"""
Question de l'utilisateur :
{question}

Documents retrouvés dans la base documentaire :
<context>
{contexte}
</context>

Réponds à la question en suivant tes consignes pédagogiques.
Explique la réponse et son raisonnement à partir du contexte disponible.
"""

    messages = [
        {
            "role": "system",
            "content": CONTEXTE_SYSTEME,
        },
        {
            "role": "user",
            "content": message_utilisateur,
        },
    ]

    texte = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    model_inputs = tokenizer(
        [texte],
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NOUVEAUX_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # On retire les tokens correspondant au prompt d'entrée.
    nouveaux_tokens = generated_ids[
        0, model_inputs.input_ids.shape[1]:
    ]

    reponse = tokenizer.decode(
        nouveaux_tokens,
        skip_special_tokens=True,
    ).strip()

    return reponse


# ---------------------------
# Programme principal
# ---------------------------

if __name__ == "__main__":

    print("Chargement du modèle d'embeddings...")
    embedding_model = SentenceTransformer(NOM_MODELE)

    print("Connexion à Qdrant...")
    client = connecter()

    tokenizer, qwen_model = charger_qwen()

    print("\nAssistant pédagogique prêt !")

    while True:
        question = input(
            "\nQuestion (Entrée pour quitter) : "
        ).strip()

        if not question:
            break

        # Étape 1 : recherche documentaire
        points = chercher(
            client,
            embedding_model,
            question,
        )

        # Étape 2 : préparation du contexte
        contexte = construire_contexte(points)

        # Affichage facultatif des résultats de Qdrant
        print("\n--- Documents retrouvés ---")

        for point in points:
            payload = point.payload or {}
            titre = payload.get("title", "")
            print(
                f"\n[{point.score:.3f}] "
                f"{payload.get('source', 'Source inconnue')} "
                f"- {titre}"
            )

        # Étape 3 : génération de la réponse
        print("\n--- Réponse pédagogique de Qwen ---\n")

        reponse = generer_reponse(
            question,
            contexte,
            tokenizer,
            qwen_model,
        )

        print(reponse)
