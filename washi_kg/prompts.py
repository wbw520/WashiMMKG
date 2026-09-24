"""Prompts for the LLM-driven pipeline stages.

Each prompt is tuned for the Washi (traditional Japanese paper) domain and asks the
model to return strict JSON so the pipeline can parse it deterministically.
"""

# ---------------------------------------------------------------------------
# Extractor — high-recall triple extraction (paper: "Triple Extraction")
# ---------------------------------------------------------------------------
EXTRACTOR_SYSTEM = """\
You are an expert annotator building a knowledge graph about Washi (traditional \
Japanese paper): its materials, fibers, tools, techniques, production steps, \
regions, workshops, history, products, and cultural context.

Given a paragraph, extract EVERY factual assertion you can as a directed triple
(head, relation, tail). Prioritise RECALL: it is better to over-generate than to
miss a relation. Capture implicit relations and varied phrasings.

Rules:
- head and tail are concise noun phrases (entities/concepts), not full sentences.
- relation is a short lowercase verb phrase with words joined by underscores
  (e.g. made_from, produced_in, used_for, part_of, requires_tool, has_property).
- Keep domain-specific terms (kozo, mitsumata, gampi, nagashizuki, suki, etc.).
- Do NOT invent facts that are not supported by the paragraph.
- Return ONLY a JSON array, no prose. Each item: {"h": ..., "r": ..., "u": ...}.
- If the paragraph contains no Washi-relevant facts, return [].
"""

EXTRACTOR_USER = """\
Paragraph:
\"\"\"
{paragraph}
\"\"\"

Return the JSON array of triples.
"""


# ---------------------------------------------------------------------------
# Detector — entity canonicalization (paper: "Entity Canonicalization")
# ---------------------------------------------------------------------------
DETECTOR_SYSTEM = """\
You resolve whether an entity mention refers to the SAME real-world concept as one
of a list of existing canonical entities in a Washi knowledge graph. Mentions may
differ by synonym, regional naming, abbreviation, or granularity
(e.g. "paper mulberry" == "kozo"; "mulberry" is broader and is NOT the same).

Return ONLY JSON: {"match": "<exact canonical name from the list>"} if the mention
denotes the same concept as one candidate, otherwise {"match": null}.
"""

DETECTOR_USER = """\
Entity mention: "{mention}"
Context (a fact it appears in): "{context}"

Existing canonical candidates:
{candidates}

Does the mention denote the SAME concept as one candidate? Reply with JSON only.
"""


# ---------------------------------------------------------------------------
# Creator — multimodal alignment (paper: "Graph Assembly", align(e))
# ---------------------------------------------------------------------------
VISION_ALIGN_SYSTEM = """\
You link an image to the single best-matching entity in a Washi knowledge graph.
You are given an image and a list of candidate entity names.

**null is the correct answer most of the time.** These images are harvested in bulk
from slide decks and web pages, so the majority depict nothing specific. Return
{"entity": null} for ALL of the following, without exception:
- logos, wordmarks, emblems, icons, or abstract decorative graphics
- slides that are mostly text, tables, charts, diagrams, or maps
- blank, near-blank, solid-colour, or heavily cropped images
- generic scenery, buildings, crowds, or people not identified as a named person
- artwork (paintings, prints) unless a candidate names that specific artwork

Answer with a candidate ONLY when the image directly and unmistakably depicts that
entity — the actual material, fiber, tool, product, workshop, process step, sheet of
paper, or named person. Ask yourself: "would a Washi curator caption this photo with
this exact entity name?" If the link rests on a loose visual association (a swirl
resembling a cloud pattern, a white object resembling paper), that is a null.
Never pick a candidate merely because it is the closest of the options offered.

Return ONLY JSON:
{"entity": "<exact candidate name or null>", "caption": "<one concise sentence describing the image>"}
"""

VISION_ALIGN_USER = """\
Candidate entities:
{candidates}

Pick the best-matching entity for this image and write a one-sentence caption.
Reply with JSON only.
"""


# ---------------------------------------------------------------------------
# Creator — abstractive A_text compression (compress_mode: llm)
# ---------------------------------------------------------------------------
COMPRESS_SYSTEM = """\
You write a single, concise, information-dense description of one entity for a Washi
(traditional Japanese paper) knowledge graph. Ground it ONLY in the provided source
text — do not add facts that are not present. Keep the entity itself as the subject,
prefer concrete domain detail over filler, and write at most {max_chars} characters.
Output ONLY the description sentence, with no preamble or quotes.
"""

COMPRESS_USER = """\
Entity: "{entity}"

Source evidence:
\"\"\"
{evidence}
\"\"\"

Write the concise description (<= {max_chars} characters).
"""
