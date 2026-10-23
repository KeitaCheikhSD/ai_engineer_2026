# Phase 0 — Préparer un corpus de texte avec Python

> Objectif : transformer des fichiers bruts (JSON, Markdown, texte) en un fichier `chunks.json` propre, prêt à être transformé en vecteurs en Phase 1.
> Niveau : débutant. Le code utilise des fonctions simples et des dictionnaires, sans classes.

---

## 1. Pourquoi cette phase existe

### Ton objectif, et une petite correction

Tu écris que, pour devenir ingénieur IA, il s'agit très souvent de faire du traitement de texte avec Python. **C'est largement vrai, avec une nuance.**

Ce qui est vrai : les modèles de langage (LLM) et les systèmes de recherche par le sens (RAG) reposent sur du texte. Avant qu'un modèle puisse l'utiliser, il faut le charger, le nettoyer, le découper, lui ajouter des informations (source, date, titre) et le stocker proprement. Ce travail s'appelle la **préparation des données**, et il occupe une grande partie du temps d'un ingénieur IA. Une règle connue le résume : *si les données d'entrée sont mauvaises, les résultats le seront aussi* (« garbage in, garbage out »).

La nuance : le métier ne se résume pas à ça. Un ingénieur IA appelle aussi des modèles via des API, mesure la qualité des réponses (évaluation), met en production, surveille les coûts et les erreurs, et travaille parfois avec des images, de l'audio ou des tableaux. Le texte est donc une compétence centrale, mais pas la seule. La Phase 0 te donne la base sur laquelle les autres phases s'appuient.

### Ce que tu vas apprendre dans cette phase

- Lire des fichiers en Python (`pathlib`, `json`)
- Manipuler des structures de données (listes, dictionnaires)
- Nettoyer du texte (expressions régulières, Unicode)
- Découper un texte intelligemment (chunking)
- Écrire un résultat structuré et **reproductible** (même résultat à chaque exécution)

---

## 2. Vue d'ensemble : le chemin du texte

```
Fichiers bruts          Nettoyage             Découpage              Sauvegarde
 (JSON, .md, .txt)  →   (espaces, Unicode) →  (chunks de ~800 car.) → chunks.json
      ÉTAPE 1              ÉTAPE 2               ÉTAPE 3 + 4            ÉTAPE 5
```

Cinq étapes, cinq fonctions. Chacune fait une seule chose.

---

## 3. Vocabulaire à connaître

| Mot | Définition simple |
|---|---|
| **Corpus** | L'ensemble des textes que tu rassembles (ici le dossier `corpus/`) |
| **Document** | Un texte d'origine : un fichier, une page web, une question du QCM |
| **Chunk** | Un morceau de document, assez court pour être vectorisé et assez complet pour être compris seul |
| **Métadonnées** | Des informations *sur* le texte : source, titre, date, thème |
| **Normalisation** | Rendre l'écriture uniforme (même apostrophe, mêmes espaces) |
| **Regex** | Expression régulière : un motif pour chercher ou remplacer du texte |
| **Id déterministe** | Un identifiant calculé à partir du contenu : même contenu, même id, à chaque fois |
| **JSON** | Un format texte pour stocker des listes et des dictionnaires |

### Pourquoi découper en chunks ?

Un modèle d'embedding (Phase 1) transforme un texte en **un seul vecteur**. Si tu lui donnes un livre entier, le vecteur est une moyenne de tout et ne ressemble à aucun passage précis. En donnant un vecteur à chaque chunk, tu pourras ensuite retrouver *le passage* qui répond à une question. Les modèles d'embedding ont aussi une longueur maximale d'entrée : ce qui dépasse est ignoré.

---

## 4. Mise en place

### Installation

```bash
pip install langchain-text-splitters
```

Le reste utilise uniquement la bibliothèque standard de Python (`json`, `re`, `hashlib`, `unicodedata`, `pathlib`).

`langchain-text-splitters` est un petit paquet qui ne contient que des outils de découpage de texte. On ne l'utilise que pour ça.

### Structure des fichiers

```
mon_projet/
├── corpus/
│   ├── data-gouv-qcm-civique-resident.json
│   └── exemple_laicite.md
├── prepare_corpus_simple.py     ← le script de ce document
└── chunks_simple.json           ← le fichier produit
```

---

## 5. Le code, étape par étape

Le script complet est dans `prepare_corpus_simple.py`. Voici chaque partie expliquée.

### Les imports et les réglages

```python
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

DOSSIER_CORPUS = Path("corpus")
FICHIER_SORTIE = Path("chunks_simple.json")
TAILLE_MAX = 800        # taille maximale d'un chunk, en caractères
CHEVAUCHEMENT = 100     # caractères répétés entre deux chunks voisins
TAILLE_MIN = 30         # on jette les chunks plus courts que ça
```

- `import` charge des outils. `json` lit/écrit du JSON, `re` gère les regex, `hashlib` calcule des empreintes, `unicodedata` normalise les caractères, `pathlib.Path` manipule les chemins de fichiers.
- Les réglages sont écrits **en majuscules, en haut du fichier** : c'est une convention pour les valeurs qu'on pourra modifier. Tu changes un chiffre ici, et tout le script suit.
- `800` caractères représentent environ 200 tokens (un token vaut grossièrement 3 à 4 caractères en français). C'est un point de départ, pas une vérité : tu testeras d'autres valeurs.

---

### Étape 1 : charger les fichiers

On veut obtenir une **liste de documents**. Chaque document est un dictionnaire avec les mêmes clés, peu importe sa source. C'est la règle la plus importante de cette étape : la suite du code n'a pas besoin de savoir d'où vient le texte.

```python
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
```

Ligne par ligne :

- `chemin.read_text(encoding="utf-8")` lit le fichier en texte. **Précise toujours `utf-8`**, sinon les accents peuvent être mal lus selon ton système.
- `json.loads(...)` transforme le texte JSON en objets Python (ici un dictionnaire).
- Le JSON du QCM est **imbriqué** : `data` contient des thèmes, chaque thème contient des questions. D'où les **deux boucles `for`** l'une dans l'autre.
- `"\n".join(...)` colle plusieurs textes en une seule chaîne, séparés par des retours à la ligne. La notation `"- " + r for r in ...` ajoute un tiret devant chaque réponse.
- Le `f"..."` (f-string) insère des variables dans un texte : `{bloc['theme']}` est remplacé par le thème.
- **Pourquoi mettre la question et les réponses ensemble ?** Une réponse seule comme « Citoyens » est incompréhensible. Avec sa question, elle a un sens. C'est l'idée d'un chunk **autonome**.
- `"meta"` garde des informations utiles pour plus tard (le thème, le numéro de la question).

```python
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
```

- `dossier.glob("*.md")` liste tous les fichiers qui finissent par `.md`. C'est ton **traitement par lot** (batch) : tu peux déposer 3 ou 300 fichiers, le code est le même.
- `sorted(...)` met les fichiers dans un ordre fixe. Sans lui, l'ordre peut changer d'un ordinateur à l'autre, et ton résultat ne serait pas reproductible.
- `chemin.name` donne `exemple.md`, `chemin.stem` donne `exemple` (sans l'extension).

---

### Étape 2 : nettoyer le texte

```python
def nettoyer(texte):
    texte = unicodedata.normalize("NFKC", texte)   # unifie les caractères Unicode
    texte = texte.replace("’", "'")                # apostrophe courbe -> droite
    texte = re.sub(r"<[^>]+>", " ", texte)         # supprime les balises HTML
    texte = re.sub(r"[ \t]+", " ", texte)          # plusieurs espaces -> un seul
    texte = re.sub(r" *\n *", "\n", texte)         # espaces autour des retours ligne
    texte = re.sub(r"\n{3,}", "\n\n", texte)       # trop de lignes vides -> une seule
    return texte.strip()
```

Chaque ligne règle un problème précis :

| Ligne | Problème réglé | Exemple |
|---|---|---|
| `normalize("NFKC", ...)` | Un même caractère peut s'écrire de plusieurs façons en Unicode. Cette fonction les unifie. | `ﬁ` devient `fi`, `…` devient `...`, l'espace insécable devient un espace normal |
| `replace("’", "'")` | Deux apostrophes visuellement proches mais différentes pour l'ordinateur | `l’examen` et `l'examen` deviennent identiques |
| `re.sub(r"<[^>]+>", " ", ...)` | Les balises HTML polluent le texte | `a<br>b` devient `a b` |
| `re.sub(r"[ \t]+", " ", ...)` | Espaces et tabulations en trop | `bonjour   le` devient `bonjour le` |
| `re.sub(r" *\n *", "\n", ...)` | Espaces qui traînent autour des retours à la ligne | `fin. \n  Début` devient `fin.\nDébut` |
| `re.sub(r"\n{3,}", "\n\n", ...)` | Beaucoup de lignes vides d'affilée | 5 lignes vides deviennent 1 |
| `.strip()` | Espaces et retours à la ligne au début et à la fin | |

**Lire les regex (version débutant) :**

- `re.sub(motif, remplacement, texte)` signifie « remplace tout ce qui correspond au motif ».
- `<[^>]+>` : un `<`, puis un ou plusieurs caractères qui ne sont pas `>`, puis un `>`.
- `[ \t]+` : un ou plusieurs espaces ou tabulations.
- `\n{3,}` : trois retours à la ligne ou plus.
- Le `r` devant les guillemets (`r"..."`) dit à Python de ne pas interpréter les `\` lui-même.

**Pourquoi on ne touche pas aux majuscules ni à la ponctuation ?** Les modèles d'embedding modernes les utilisent pour comprendre le texte. On en retire le moins possible. Les retours à la ligne `\n` sont conservés volontairement : l'étape 3 s'en sert pour repérer les paragraphes.

Pourquoi l'**ordre** compte : on retire les balises *avant* de réduire les espaces, car on remplace une balise par un espace, ce qui peut créer des doubles espaces.

---

### Étape 3 : découper en chunks

C'est la partie centrale. Elle fonctionne en **deux niveaux**.

```python
decoupeur = RecursiveCharacterTextSplitter(
    chunk_size=TAILLE_MAX,
    chunk_overlap=CHEVAUCHEMENT,
    separators=["\n\n", "\n", ". ", " ", ""],   # du plus gros au plus petit
    keep_separator="end",                       # le point reste à la fin de sa phrase
)
```

On crée un « découpeur » une seule fois. Ses paramètres :

- `chunk_size` : taille maximale d'un chunk.
- `chunk_overlap` : nombre de caractères répétés entre deux chunks voisins (voir plus bas).
- `separators` : les endroits où il a le droit de couper, **du meilleur au pire** : d'abord entre deux paragraphes (`\n\n`), sinon entre deux lignes (`\n`), sinon entre deux phrases (`. `), sinon entre deux mots (` `), et en tout dernier recours entre deux lettres (`""`).
- `keep_separator="end"` : quand il coupe sur un point, le point reste avec sa phrase au lieu de partir au début du morceau suivant.

```python
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
```

**Niveau 1 : par sections.**
`re.split(r"\n(?=#{1,6} )", texte)` coupe le texte à chaque titre Markdown (`#`, `##`, `###`...). Le motif signifie : « un retour à la ligne, *seulement s'il est suivi* de 1 à 6 `#` puis d'un espace ». Le `(?=...)` veut dire « regarde ce qui suit sans le consommer », ce qui garde le titre dans la section qu'il ouvre. Résultat : **un titre n'est jamais séparé de son contenu**.

**Niveau 2 : par taille.**
Si une section tient dans 800 caractères, on la garde telle quelle. Sinon :

1. `section.partition("\n")` sépare la première ligne (le titre) du reste.
2. `decoupeur.split_text(corps)` découpe le reste en respectant les paragraphes, puis les phrases.
3. On **recolle le titre devant chaque morceau**. Ainsi, même le 3e morceau d'une longue section sait de quoi il parle.

Détail à connaître : le titre est ajouté *après* le découpage, donc un chunk peut dépasser `TAILLE_MAX` de la longueur du titre. Ce n'est pas grave, `TAILLE_MAX` est une cible, pas une limite stricte pour le modèle.

**Le chevauchement (`CHEVAUCHEMENT`), en image :**

```
Chunk 1 : [ phrase A  phrase B  phrase C ]
Chunk 2 :                       [ phrase C  phrase D  phrase E ]
                                  ↑ répétée : une idée coupée à la frontière apparaît dans les deux chunks
```

Il ne s'applique qu'à l'intérieur d'une même section longue. Entre deux sections différentes, il n'y a pas de chevauchement : chaque section est autonome.

---

### Étape 4 : fabriquer les chunks (id et métadonnées)

```python
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
```

- **`fabriquer_id`** calcule une empreinte (`sha256`) à partir de la source, de la position et du texte, et garde les 16 premiers caractères. C'est ce qui rend l'id **déterministe** : si tu relances le script, tu obtiens exactement les mêmes ids. Si le texte d'un document change, l'id change aussi. Un id aléatoire (type UUID) ou un simple compteur ne te le permettrait pas.
- **`enumerate(textes)`** donne à la fois la position (0, 1, 2...) et le texte.
- **`continue`** saute au chunk suivant : un chunk de moins de 30 caractères (par exemple un titre seul) n'apporte rien à la recherche, on l'ignore.
- **`**document["meta"]`** copie les métadonnées propres au document (thème, numéro de question) dans le dictionnaire des métadonnées du chunk.
- `"date": None` : la date n'existe pas dans nos sources. On garde la clé pour que tous les chunks aient la même forme.

---

### Étape 5 : sauvegarder en JSON

```python
def sauvegarder(chunks, chemin):
    sans_doublon = {c["id"]: c for c in chunks}     # même id = une seule entrée
    liste = list(sans_doublon.values())
    chemin.write_text(json.dumps(liste, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(liste)
```

- `{c["id"]: c for c in chunks}` crée un dictionnaire dont la **clé est l'id**. Dans un dictionnaire, une clé est unique : deux chunks identiques s'écrasent, donc plus de doublons.
- `json.dumps(..., ensure_ascii=False, indent=2)` écrit du JSON lisible. `ensure_ascii=False` garde les accents tels quels au lieu de les remplacer par des codes `\u00e9`.
- Le fichier est **réécrit en entier** à chaque exécution. Relancer le script ne rajoute donc rien : c'est le deuxième mécanisme anti-doublons.

---

### Le programme principal

```python
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
```

- `if __name__ == "__main__":` signifie « exécute ce bloc seulement quand on lance ce fichier directement » (et pas quand un autre fichier l'importe).
- L'ordre des opérations est celui du schéma du début : **charger, nettoyer, découper (dans `fabriquer_chunks`), sauvegarder**.
- `DOSSIER_CORPUS / "fichier.json"` : l'opérateur `/` de `pathlib` assemble des chemins proprement, sur tous les systèmes.

---

## 6. Lancer le script et lire le résultat

```bash
python prepare_corpus_simple.py
```

Sortie obtenue sur les fichiers de ce projet :

```
210 documents lus -> 213 chunks écrits dans chunks_simple.json
```

- 210 documents = 209 questions du QCM + 1 fichier Markdown.
- 213 chunks = 209 pour le QCM + 4 pour le fichier Markdown (le titre principal seul a été jeté car trop court).

Un chunk du QCM dans `chunks_simple.json` :

```json
{
  "id": "f02e2b82dfd4c038",
  "text": "Thème : Principes et valeurs de la République\nQuestion : Parmi les propositions suivantes, laquelle constitue une participation citoyenne ?\nRéponses :\n- S'engager dans une association (de quartier, humanitaire...).\n- Organiser une collecte pour une association humanitaire.\n- Voter aux élections.",
  "source": "data-gouv-qcm-civique-resident.json",
  "metadata": {
    "title": "Parmi les propositions suivantes, laquelle constitue une participation citoyenne ?",
    "date": null,
    "chunk_index": 0,
    "theme": "Principes et valeurs de la République",
    "question_id": 1
  }
}
```

**Vérification à faire toi-même :** lance le script deux fois et compare les deux fichiers produits. Ils doivent être identiques. C'est la preuve que ton traitement est reproductible.

---

## 7. Les trois questions de ton énoncé

### Si je découpe au milieu d'une phrase, qu'est-ce que ça casse ?

Deux choses :

1. **La vectorisation** : un fragment sans sujet ni verbe (« …ont instauré une école publique, gratuite, ob ») donne un vecteur de mauvaise qualité, parce que le modèle n'a pas le sens complet.
2. **L'affichage** : quand ce chunk sera retourné à l'utilisateur comme résultat, il verra un texte coupé et incompréhensible.

C'est pourquoi on coupe d'abord aux titres, puis aux paragraphes, puis aux phrases, et seulement en dernier recours entre deux mots.

Pour voir la différence, essaie ce découpage brutal sur ton fichier :

```python
texte = open("corpus/exemple_laicite.md", encoding="utf-8").read()
morceaux = [texte[i:i + 300] for i in range(0, len(texte), 300)]
print(morceaux[1])   # commence et finit au milieu d'une phrase
```

### Comment gérer les documents très courts et très longs ?

- **Très courts** (une question du QCM : environ 230 caractères) : un seul chunk, aucun découpage. Un seuil minimal (`TAILLE_MIN`) écarte les fragments inutiles, comme un titre seul.
- **Très longs** : découpage par sections, puis par taille. Le chevauchement limite la perte de contexte aux frontières.
- **Le dernier chunk d'un document** est souvent plus petit que les autres. C'est normal, et le seuil minimal évite seulement les cas extrêmes.

### Comment ne pas dupliquer au relancement ?

Trois mécanismes combinés :

1. **Id déterministe** : même contenu, même id.
2. **Dictionnaire par id** : deux chunks de même id s'écrasent (étape 5).
3. **Fichier réécrit en entier** : on ne fait jamais « ajouter à la suite de l'ancien ».

Si un document source change, ses nouveaux chunks ont de nouveaux ids, et les anciens disparaissent parce que le fichier est reconstruit.

---

## 8. Expériences à faire pour comprendre

Change un réglage, relance, compare. C'est la meilleure façon d'apprendre.

| Expérience | Ce que tu observes |
|---|---|
| `TAILLE_MAX = 300` | Beaucoup plus de chunks, plus courts. Les passages perdent leur contexte. |
| `TAILLE_MAX = 3000` | Presque un chunk par section. Les vecteurs seront plus vagues. |
| `CHEVAUCHEMENT = 0` puis `200` | Compare la fin d'un chunk et le début du suivant dans une longue section. |
| Ajoute un fichier `.txt` de plusieurs pages dans `corpus/` | Vérifie qu'il est lu automatiquement, puis lis ses chunks : sont-ils compréhensibles seuls ? |
| Supprime la ligne `texte.replace("’", "'")` | Compte les apostrophes différentes dans ton résultat. |

Pour compter les chunks et leur taille :

```python
import json, statistics
chunks = json.load(open("chunks_simple.json", encoding="utf-8"))
tailles = [len(c["text"]) for c in chunks]
print(len(chunks), min(tailles), statistics.mean(tailles), max(tailles))
```

---

## 9. Limites connues de cette version

Un bon exercice : repérer ce qui n'est pas parfait.

1. **Le titre principal (`# ...`) est perdu.** Il est seul dans sa section, donc trop court, donc ignoré. Les chunks gardent leur titre de section (`##`), mais pas le titre du document. On pourrait l'ajouter en tête de chaque chunk.
2. **`chunk_index` commence à 1 pour le fichier Markdown**, car le chunk 0 (le titre seul) a été jeté après avoir été numéroté. Ce n'est pas un bug grave, mais c'est à savoir.
3. **Seulement des fichiers `.json`, `.md`, `.txt`.** Les PDF (`pypdf` ou `pdfplumber`) et les pages web (`requests` + `beautifulsoup4`) restent à ajouter : il suffira d'écrire une fonction `charger_pdf` et une fonction `charger_web` qui renvoient des documents **au même format**.
4. **Le nettoyage HTML est sommaire.** Pour du vrai HTML, on utilisera BeautifulSoup (`soup.get_text()`), qui gère aussi `&amp;` et `&nbsp;`.
5. **Taille en caractères, pas en tokens.** C'est suffisant pour démarrer. En Phase 1, tu pourras vérifier avec le tokenizer du modèle choisi.
6. **Aucune détection de doublons « presque identiques ».** Seuls les textes strictement identiques (même source, même position) sont fusionnés.

---

## 10. Annexe A : comprendre le découpage avec du Python pur

Si tu veux voir ce qu'il y a « sous le capot », voici une version minimale qui regroupe des paragraphes jusqu'à une taille maximale :

```python
def decouper_simple(texte, taille_max=800):
    paragraphes = texte.split("\n\n")        # 1. on coupe aux paragraphes
    chunks = []
    courant = ""                             # 2. le chunk en cours de construction
    for p in paragraphes:
        if len(courant) + len(p) + 2 <= taille_max:
            courant = (courant + "\n\n" + p).strip()   # ça rentre : on ajoute
        else:
            if courant:
                chunks.append(courant)       # ça ne rentre pas : on range le chunk
            courant = p                      # et on en commence un nouveau
    if courant:
        chunks.append(courant)               # 3. on n'oublie pas le dernier
    return chunks
```

Essaie-la sur `exemple_laicite.md` : tu obtiens 4 chunks de 452, 779, 585 et 359 caractères. Regarde le résultat de près, tu verras **deux défauts** :

- Un titre (`## Une histoire en plusieurs étapes`) se retrouve **à la fin d'un chunk** alors que son contenu est dans le suivant. Cet « orphelin » est exactement ce que le découpage par sections de l'étape 3 évite.
- Un paragraphe de plus de 800 caractères resterait entier, car cette version ne sait pas le recouper en phrases.

Ces deux défauts expliquent pourquoi on utilise `RecursiveCharacterTextSplitter` (qui gère les phrases et le chevauchement) et pourquoi on coupe d'abord par sections.

---

## 11. Annexe B : la version orientée objet

Le dépôt contient aussi `prepare_corpus.py`, qui fait le même travail avec des classes (`Document`, `Chunk`, `TextCleaner`, `QcmJsonLoader`, `SectionChunker`, `ChunkStore`, `CorpusPipeline`). Elle est plus facile à étendre quand on ajoute des sources (PDF, web), mais plus longue à lire.

**Conseil :** comprends d'abord la version simple de ce document. Passe à la version objet quand tu as besoin d'ajouter plusieurs sources de types différents.

---

## 12. Lien avec la Phase 1

Ce que la Phase 0 t'a produit : `chunks.json`, une liste de chunks avec `id`, `text`, `source` et `metadata`.

Ce que la Phase 1 fera avec :

1. Charger `chunks.json`.
2. Charger un modèle d'embedding avec `sentence-transformers`.
3. Transformer chaque `text` en vecteur : `vecteurs = modele.encode(textes)`.
4. Garder le **même ordre** entre les chunks et les vecteurs (la ligne `i` de la matrice correspond au chunk `i`).
5. Sauvegarder les vecteurs (par exemple avec `numpy.save`) à côté de `chunks.json`.

Le schéma complet, de la Phase 0 à la recherche :

```
Phase 0                          Phase 1                        Recherche
fichiers → chunks.json   →   chunks → vecteurs   →   question → vecteur → chunk le plus proche
```

---

## 13. Checklist de fin de phase

- [ ] Je sais expliquer ce qu'est un chunk et pourquoi on découpe.
- [ ] Je peux lancer le script et obtenir `chunks_simple.json`.
- [ ] Je l'ai lancé deux fois et le fichier est identique.
- [ ] J'ai changé `TAILLE_MAX` et `CHEVAUCHEMENT` et observé la différence.
- [ ] J'ai ajouté un fichier à `corpus/` et vérifié qu'il est traité.
- [ ] Je sais expliquer pourquoi un id déterministe évite les doublons.
- [ ] Je peux nommer deux défauts de ma version actuelle (section 9).
- [ ] Bonus : j'ai écrit `charger_pdf` avec `pypdf` ou `pdfplumber`.
- [ ] Bonus : j'ai écrit `charger_web` avec `requests` + `beautifulsoup4`.
