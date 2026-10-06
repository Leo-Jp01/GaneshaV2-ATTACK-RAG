import json
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer


def cargar_chunks(path):
    """Lee el JSON generado por el parser y devuelve la lista de chunks."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def generar_embeddings(chunks, model):
    """Añade el prefijo 'passage:' y embebe todos los chunks en batch."""
    textos = ["passage: " + chunk["text"] for chunk in chunks]
    return model.encode(textos, batch_size=32, show_progress_bar=True)


def crear_coleccion(client, nombre="mitre_attack", dim=768):
    """Crea la colección en Qdrant si no existe ya."""
    client.create_collection(
        collection_name=nombre,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )


def armar_points(chunks, embeddings):
    """Combina cada chunk con su embedding correspondiente en un PointStruct."""
    return [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i].tolist(),
            payload=chunks[i]
        )
        for i in range(len(chunks))
    ]


def buscar(pregunta, model, client, top_k=5, umbral=0.75):
    """
    Embebe la pregunta del usuario y busca los chunks más similares en Qdrant,
    aplicando top-K y un umbral mínimo de similitud (los dos mecanismos que
    diseñamos: nunca traer más de top_k, pero descartar lo que no llegue al umbral).
    """
    texto_query = "query: " + pregunta
    vector_pregunta = model.encode(texto_query).tolist()

    resultados = client.query_points(
        collection_name="mitre_attack",
        query=vector_pregunta,
        limit=top_k,
        score_threshold=umbral
    )

    return resultados.points


def expandir_si_es_padre(resultado_principal, client, umbral=0.75):
    """
    Si el resultado con mayor score es una técnica padre (no sub-técnica)
    y supera el umbral, trae TODAS sus sub-técnicas por filtro de metadata
    (no por similitud semántica) -- garantiza completitud de la familia.
    """
    payload = resultado_principal.payload

    if payload["is_subtechnique"] or resultado_principal.score < umbral:
        return []  # no es padre, o no pasó el umbral -> no expandimos

    from qdrant_client.models import Filter, FieldCondition, MatchValue

    hijas = client.scroll(
        collection_name="mitre_attack",
        scroll_filter=Filter(
            must=[FieldCondition(key="parent_technique_id", match=MatchValue(value=payload["technique_id"]))]
        ),
        limit=50
    )

    return hijas[0]  # scroll devuelve (points, next_page_offset); solo nos interesan los points


def resumen_corto(texto, max_chars=200):
    """Corta hasta el primer punto, con un tope máximo por si la oración es muy larga."""
    primer_punto = texto.find(".")
    if primer_punto != -1 and primer_punto < max_chars:
        return texto[:primer_punto + 1]
    return texto[:max_chars] + "..."

def buscar(pregunta, model, client, top_k=5, umbral=0.75):
    texto_query = "query: " + pregunta
    vector_pregunta = model.encode(texto_query).tolist()

    resultados = client.query_points(
        collection_name="mitre_attack",
        query=vector_pregunta,
        limit=top_k,
        score_threshold=umbral
    ).points

    if not resultados:
        print("⚠️  No se encontró información con suficiente confianza. Intentá reformular la pregunta con más contexto técnico.")
        return

    for r in resultados:
        print(f"[{r.score:.3f}] {r.payload['technique_id']} - {r.payload['technique_name']}")
        print(f"   {resumen_corto(r.payload['text'])}")

    hijas = expandir_si_es_padre(resultados[0], client, umbral)
    if hijas:
        print(f"\n📎 Sub-técnicas de {resultados[0].payload['technique_name']}:")
        for h in hijas:
            print(f"   - {h.payload['technique_id']}: {h.payload['technique_name']}")

def main():
    model = SentenceTransformer("intfloat/multilingual-e5-base")
    client = QdrantClient("localhost", port=6333)

    with open("chunks_output.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)
    
    for c in chunks:
        if c["technique_id"] == "T1110":
            print(c["text"])

    #chunks = cargar_chunks("chunks_output.json")
    #embeddings = generar_embeddings(chunks, model)

    #crear_coleccion(client)
    #points = armar_points(chunks, embeddings)

    #client.upsert(collection_name="mitre_attack", points=points)
    #print(f"Subidos {len(points)} points. Shape embeddings: {embeddings.shape}")

    resultados = buscar("¿Cómo se detecta la fuerza bruta en inicios de sesion?", model, client, umbral=0.60)
    buscar("Brute Force", model, client, umbral=0.60)                          # inglés, coincide casi literal con el nombre
    buscar("fuerza bruta", model, client, umbral=0.60)                          # español, término directo
    buscar("¿Cómo se detecta la fuerza bruta en inicios de sesion?", model, client, umbral=0.60)  # la que ya probaste, con la palabra "detecta"

    for r in resultados:
        print(f"[{r.score:.3f}] {r.payload['technique_id']} - {r.payload['technique_name']}")

if __name__ == "__main__":
    main()
