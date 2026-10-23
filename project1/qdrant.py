from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Document

COLLECTION = "items"

# connect to Qdrant Cloud
client = QdrantClient(
    url="https://f7298877-b4bd-4910-98ed-f11b579cef97.eu-central-1-0.aws.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6OTAwMDg3YTEtMWUzOS00MWZkLTg3M2MtMDMyNjQ1YTU2Y2Q1In0.3NvOvlgKWX9dHssGFQIEoUSl7yGj3vwhm-nsqhMO9Bc",
    cloud_inference=True
)

# 2. Créer la collection seulement si elle n'existe pas
if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# 3. Charger le modèle d'embeddings
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 4. Lire le fichier JSON
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Si le fichier contient un seul chunk (dict), on le met dans une liste
if isinstance(chunks, dict):
    chunks = [chunks]

# 5. Générer tous les vecteurs d'un coup
texts = [c["text"] for c in chunks]
embeddings = model.encode(texts, show_progress_bar=True)

# 6. Construire les points
points = [
    PointStruct(
        # Qdrant exige un entier ou un UUID : on dérive un UUID stable de l'id d'origine
        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, str(c["id"]))),
        vector=emb.tolist(),
        payload={
            "chunk_id": c["id"],
            "text": c["text"],
            "source": c.get("source"),
            **c.get("metadata", {}),
        },
    )
    for c, emb in zip(chunks, embeddings)
]

# 7. Insérer par lots
client.upload_points(collection_name=COLLECTION, points=points, batch_size=64)

print(f"{len(points)} chunks enregistrés dans Qdrant !")
print("Dimensions du vecteur :", len(embeddings[0]))