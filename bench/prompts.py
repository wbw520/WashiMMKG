"""Prompts for turning a computed benchmark item into natural language.

The gold answer already exists before any of these run. The model's job is strictly
verbalization: phrase the question a curator would ask to elicit that answer, and
state the answer in prose. It is never asked what the answer *is*, so it cannot move
it — `validate.py` then checks that every gold atom survived.

Two failure modes these prompts exist to prevent:

* **Leakage.** The obvious phrasing of a computed item embeds its own answer
  ("Which paper made from kozo by Iwano Ichibei ...?" when the answer is the paper).
  Each template is told which strings are the answer and forbidden to use them.
* **Invention.** Given a sparse spec a model will pad the answer with plausible Washi
  facts that are not in the graph. Anything not in the spec is off-limits, because the
  retrieval metrics score against `gold_triples` and nothing else.
"""

SYSTEM = """\
You write evaluation items for a benchmark on Washi (traditional Japanese paper).

You are given a REASONING STRUCTURE already extracted from a knowledge graph, together
with its ANSWER. Your only task is to express them in natural language.

Hard rules:
- The answer is fixed. Do not change, extend, or second-guess it.
- Use ONLY facts present in the given structure. Never add Washi knowledge of your own,
  however plausible - unsupported detail makes the item unscoreable.
- The QUESTION must not contain any answer string, nor an obvious paraphrase of one.
  A reader must not be able to answer it from its own wording.
- The QUESTION must genuinely require the whole structure. If one lookup settles it,
  rewrite it so every part of the structure is needed.
- Write as a curious person would ask, not as a graph traversal. These words are BANNED
  from the question: entity, entities, node, triple, graph, relation, record, dataset,
  knowledge base, item. Name the kind of thing instead - "which paper", "which workshop",
  "which papermaker", "which technique", "which region". If you cannot tell what kind of
  thing it is, "which one" is still better than "which entity".
- The ANSWER states the answer and briefly why, in 30-60 words. No preamble.

Return ONLY JSON: {"question": "...", "answer": "..."}
"""

_TAIL = """
ANSWER (must be conveyed, and must NOT appear in the question): {answer}

Return the JSON.
"""

CHAIN = """\
A path was followed through Washi knowledge, one step at a time:
{path}

Ask what stands at the end of that path, starting from "{start}".

Name ONLY "{start}". Every other thing on the path -- the intermediate results and the
endpoint -- must be referred to by what it is, never by its name: "the workshop that makes
it", "the region that contains it". Naming an intermediate hands the reader the route and
collapses a multi-step question into a single lookup, which is the one thing this item is
built to measure.

Say it the way a person would. Not naming the waypoints tempts you into stacking relative
clauses -- "the prefecture containing the city where the town that X is based in sits" --
and past two steps nobody talks like that. Two ways out, both of them things people
actually say:

  Ask the reader to follow the trail.
    "Start from Masuda Takuto and follow where he is based up to the prefecture. Which
     prefecture is it?"

  Or break the sentence in two, setting the scene first and asking at the end.
    "Masuda Takuto works in a hot-spring town on the Izu peninsula. Which prefecture is
     that town in?"

The second reads better and is preferred where the path is short enough to carry it. Use
the first for the long ones. Either way: at most one relative clause, and no "which in
turn", no "that contains the ... where ...".
""" + _TAIL

INTERSECTION = """\
Exactly one thing (or a few) satisfies BOTH of these at once:
{constraints}

Ask which thing satisfies both. Both conditions must be stated - either one alone
leaves several possibilities.
""" + _TAIL

COMPARATIVE = """\
Two things of the same kind ({cls}) differ in one respect:
  {left}  ->  {relation}: {left_values}
  {right} ->  {relation}: {right_values}

Ask how the two compare in that respect. Name both things and the respect being
compared; do not state either of their values.
""" + _TAIL

TEMPORAL = """\
"{entity}" is first dated to {year}, and the record attaches to it:
{developments}{makers}

Ask what is known about the history of "{entity}".

The question MUST name "{entity}" itself - it is the subject being asked about, not the
answer. Do NOT invert the item by describing the facts below and asking which thing they
belong to: those facts ARE the answer, so a question built from them answers itself.
State none of them, and not the date either.
""" + _TAIL

AGGREGATION_ABSENCE = """\
Among the things that satisfy "{pop_rel}: {pop_obj}" ({pop_size} of them), some also
satisfy "{exc_rel}: {exc_obj}" - and some do not.

Ask which ones do NOT. The point of the question is the absence: what is missing is the
answer. State both conditions; list neither group.
""" + _TAIL

AGGREGATION_EXTREMUM = """\
Computed over the whole body of Washi knowledge: {operation}.
Result: {answer_value}{extra}

Ask for that superlative. Make clear what is being ranked and over what, so the reader
knows a single record cannot settle it - the whole collection must be surveyed.
""" + _TAIL

MULTIMODAL_IDENTIFY = """\
A photograph accompanies this question. It shows: {subject}
What the record says about it:
{facts}

Ask what the photograph shows, and what is known about it.

**The question must not name "{subject}", nor any of the facts above, nor any word that
would give the name away.** Refer to what is pictured only as what it visibly is - "the
paper in this photograph", "the object shown here", "the workshop pictured". Identifying
it IS the task; a question that names it has no task left. Keep it natural: a visitor
holding a photograph and asking what they are looking at.
""" + _TAIL

MULTIMODAL_ATTRIBUTE = """\
A photograph accompanies this question. It shows: {subject}
The fact being asked for: {subject} -- {relation} -> {value}

Ask for that property of the thing in the photograph.

**The question must not name "{subject}" and must not state "{value}".** Refer to the
subject only as what it visibly is ("the paper shown here", "the object in this
photograph"). The reader has to recognise it from the image first and then recall the
property, so naming it removes half the question.
""" + _TAIL

MULTIMODAL_GROUNDED = """\
A photograph of {subject} accompanies this question.
The fact being asked for: {subject} -- {relation} -> {value}

Ask that property of {subject}, mentioning that a photograph of it is provided. Naming
the subject is correct here - the reader knows what they are looking at and wants the
property. Do not state "{value}".
""" + _TAIL

MM_CHAIN = """\
In the knowledge base, the thing at the END of this path carries a photograph:
{path}

What that photograph shows: {caption}

Ask the reader to follow the path from "{start}" and then to say what the picture of what
they arrive at shows. The picture is NOT attached to the question -- it hangs on the
endpoint, and is seen only by someone who has walked there. So write "its photograph",
"the picture of that paper", "how it looks in the archive's photograph" -- never "the
accompanying photograph", "the photograph below", "the provided image", or "this
photograph", all of which promise the reader a picture that is not in front of them.

Name ONLY "{start}". Refer to every later step by its role -- "the paper it produces",
"the maker of that paper" -- never by name. Naming any of them removes the walk.

**Name "{start}" and the connections, but never name the endpoint, and never state
anything from the description above.** The description is the answer: it records what the
picture looks like, which no fact in the path records. A reader with the path but without
the picture must be able to reach the right thing and still not be able to answer.

Ask about the thing, not about the photograph. "What does its photograph show?" is how a
database would put it; a person asks what the fiber looks like, what colour the sheet is,
how the surface is worked -- and mentions the archive's picture only as where to look:

  "Trace the paper Zhang Daqian used back to the fiber it is made from. What does that
   fiber look like at the pulp stage, as the archive photograph of it shows?"

That asks for the same content, is answerable only by someone who reached the endpoint and
looked, and sounds like a question rather than a lookup.
""" + _TAIL
