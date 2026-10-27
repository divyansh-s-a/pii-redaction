#!/usr/bin/env python3
"""Generate spoken aliases for a mapping, once, with a model.

A regex covers how a name is WRITTEN in documents; speech transcribed in Hindi
or Marathi spells it in Devanagari, splits it ("zor vex"), mis-hears it, or
spells an abbreviation out letter by letter. This asks a model (Gemini, via
common.generate) for those forms per entry and writes them to
<mapping>.aliases.yaml next to the mapping — never into this repository. The
detector reads the sidecar automatically; an entry whose pattern changes
loses its stale aliases (each is keyed by the pattern's hash).

  python3 tools/expand_mapping.py MAPPING.yaml [--languages hi,mr,en]
      [--force]
Aliases that collide with an invented name, or are shorter than 3
characters, are dropped.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import common
import entity_map

PROMPT = """For each named entity below (a brand, product, organisation, agency or
a distinctive phrase about one), list how it is likely to appear in an
automatic transcript of people speaking {langs}: the canonical spoken name;
Devanagari spellings (Hindi and Marathi); common Latin transliterations and
mis-spellings a speech recogniser produces; split or joined forms; for an
abbreviation, the letters spelled out as spoken. Do not include generic words
on their own, other brands, or translations of the meaning. At most 12
aliases per entity.

ENTITIES (id, category, regex, known forms):
{entities}
"""
SCHEMA = {
    "type": "OBJECT",
    "properties": {"entities": {"type": "ARRAY", "items": {
        "type": "OBJECT",
        "properties": {"id": {"type": "STRING"},
                       "name": {"type": "STRING"},
                       "aliases": {"type": "ARRAY",
                                   "items": {"type": "STRING"}}},
        "required": ["id", "name", "aliases"]}}},
    "required": ["entities"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mapping")
    ap.add_argument("--languages", default="hi,mr,en")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    path = Path(a.mapping).resolve()
    if ROOT in path.parents:
        sys.exit("refusing: a mapping holds real names and must live outside "
                 "this repository")
    em = entity_map.load(path)
    side = path.with_name(path.stem + ".aliases.yaml")
    import yaml
    old = (yaml.safe_load(side.read_text()) or {}).get("entries", {}) \
        if side.exists() and not a.force else {}
    todo = [e for e in em.entries
            if e["render"] == "voice" or e["category"] != "PERSON"]
    todo = [e for e in todo if e["id"] not in old or old[e["id"]].get(
        "pattern_sha256") != hashlib.sha256(e["pattern"].encode()).hexdigest()]
    if not todo:
        print("aliases up to date")
        return
    invented = {t.lower() for t in em.invented_terms()}
    out = dict(old)
    model = common.resolve_model()
    for b in range(0, len(todo), 15):
        chunk = todo[b: b + 15]
        listing = "\n".join(f"{e['id']}\t{e['category']}\t{e['pattern']}\t"
                            + "; ".join(e["forms"][:4]) for e in chunk)
        common.set_usage_context("expand_mapping")
        raw = json.loads(common.generate(
            model, [{"text": PROMPT.format(langs=a.languages,
                                           entities=listing)}], SCHEMA))
        for r in raw.get("entities", []):
            e = em.by_id.get(r.get("id"))
            if e is None:
                continue
            al = [x.strip() for x in r.get("aliases", [])
                  if len(x.strip()) >= 3 and x.strip().lower() not in invented]
            out[e["id"]] = {"pattern_sha256": hashlib.sha256(
                e["pattern"].encode()).hexdigest(),
                "name": r.get("name", "").strip(),
                "aliases": list(dict.fromkeys(al))[:12]}
    side.write_text(yaml.safe_dump({"entries": out}, allow_unicode=True,
                                   sort_keys=True))
    entity_map.load(path)        # validates aliases against invented names
    print(f"{len(todo)} entr(y/ies) expanded -> {side}")


if __name__ == "__main__":
    main()
