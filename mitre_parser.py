"""
mitre_parser.py

Parsea el dataset STIX de MITRE ATT&CK (enterprise-attack.json) y produce
una lista de "chunks" listos para embeber, usando chunking parent-child:
cada técnica (padre o sub-técnica) se convierte en UN chunk independiente,
enlazado a su padre/hijas por metadata (no por anidamiento en el texto).

Diseño (decidido junto con Leo antes de escribir esto):
  - Dos pasadas sobre el JSON:
      Pasada 1: indexar TODOS los attack-pattern por su external_id (T1055, T1055.011, ...)
      Pasada 2: recorrer el índice y resolver relaciones padre-hijo con seguridad
  - Si una sub-técnica referencia un parent_id que NO existe en el índice
    (dato corrupto, revocado, typo en la fuente), se registra como "huérfana"
    en vez de fallar o inventar un padre. Esto evita:
      a) que el script explote con un KeyError a mitad de 46MB de datos
      b) que se cree metadata falsa (parent_name inventado)
      c) iterar tiempo de más sobre relaciones que no existen
  - El texto de cada chunk es "name: description" (contextual prefixing),
    para que el embedding capture tanto la etiqueta como el contenido.
  - Se descartan objetos revoked=true y x_mitre_deprecated=true por defecto
    (son técnicas obsoletas/reemplazadas — ruido para un RAG de uso actual).

Uso:
    python3 mitre_parser.py enterprise-attack.json chunks_output.json
"""

import json
import sys
from dataclasses import dataclass, asdict, field


@dataclass
class TechniqueChunk:
    """Un chunk listo para embeber: texto + metadata (payload de Qdrant)."""
    technique_id: str
    technique_name: str
    text: str                      # esto es lo que se embebe: "name: description"
    is_subtechnique: bool
    parent_technique_id: str | None
    parent_technique_name: str | None
    tactics: list[str] = field(default_factory=list)
    platforms: list[str] = field(default_factory=list)
    url: str = ""


def _get_external_id(stix_obj: dict) -> str | None:
    """
    Extrae el ID tipo 'T1055' o 'T1055.011' del array external_references.
    Solo nos interesa la referencia cuyo source_name es 'mitre-attack'
    (las demás son citas bibliográficas).
    """
    for ref in stix_obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id")
    return None


def _get_url(stix_obj: dict) -> str:
    for ref in stix_obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("url", "")
    return ""


def _get_tactics(stix_obj: dict) -> list[str]:
    """kill_chain_phases -> phase_name es donde STIX guarda las tácticas."""
    return [
        phase["phase_name"]
        for phase in stix_obj.get("kill_chain_phases", [])
        if phase.get("kill_chain_name") == "mitre-attack"
    ]


def parse_mitre_attack(json_path: str) -> tuple[list[TechniqueChunk], list[str]]:
    """
    Devuelve (chunks, huerfanos).
    huerfanos: lista de technique_id cuya sub-técnica no encontró padre en el índice.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_objects = data["objects"]

    # ---------- PASADA 1: indexar todos los attack-pattern por external_id ----------
    index: dict[str, dict] = {}
    for obj in all_objects:
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked", False) or obj.get("x_mitre_deprecated", False):
            continue  # técnica obsoleta/reemplazada: fuera del índice directamente

        ext_id = _get_external_id(obj)
        if ext_id is None:
            continue  # objeto sin ID de MITRE válido, no debería pasar pero por seguridad

        index[ext_id] = obj

    # ---------- PASADA 2: construir chunks resolviendo relaciones ----------
    chunks: list[TechniqueChunk] = []
    orphans: list[str] = []

    for ext_id, obj in index.items():
        is_sub = obj.get("x_mitre_is_subtechnique", False)
        name = obj.get("name", "")
        description = obj.get("description", "").strip()

        parent_id = None
        parent_name = None

        if is_sub:
            candidate_parent_id = ext_id.split(".")[0]
            parent_obj = index.get(candidate_parent_id)

            if parent_obj is None:
                # Caso borde: el padre no existe en el índice (revocado, typo, etc.)
                # No inventamos nada. Se registra y se sigue de largo.
                orphans.append(ext_id)
            else:
                parent_id = candidate_parent_id
                parent_name = parent_obj.get("name", "")

        # Contextual prefixing: el nombre va DENTRO del texto embebido, no solo en metadata
        text = f"{name}: {description}"

        chunks.append(TechniqueChunk(
            technique_id=ext_id,
            technique_name=name,
            text=text,
            is_subtechnique=is_sub,
            parent_technique_id=parent_id,
            parent_technique_name=parent_name,
            tactics=_get_tactics(obj),
            platforms=obj.get("x_mitre_platforms", []),
            url=_get_url(obj),
        ))

    return chunks, orphans


def main():
    if len(sys.argv) != 3:
        print("Uso: python3 mitre_parser.py <enterprise-attack.json> <output.json>")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]

    chunks, orphans = parse_mitre_attack(input_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)

    print(f"✅ {len(chunks)} chunks generados -> {output_path}")
    print(f"⚠️  {len(orphans)} sub-técnicas huérfanas (padre no encontrado en el índice)")
    if orphans:
        print(f"   IDs huérfanos: {orphans}")

    n_parent = sum(1 for c in chunks if not c.is_subtechnique)
    n_sub = sum(1 for c in chunks if c.is_subtechnique)
    print(f"📊 Técnicas padre: {n_parent} | Sub-técnicas: {n_sub}")


if __name__ == "__main__":
    main()
