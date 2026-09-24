"""Prompts for the wiki-walk retriever.

Each corresponds to one decision in the search, and each is deliberately narrow. The
navigator sees one page and picks links; the verifier sees one finished path and judges
it; the answerer sees only verified evidence. Nothing is asked to do two of these at
once, because a model asked to both explore and conclude tends to conclude early.
"""

# --------------------------------------------------------------------- decomposition
DECOMPOSE_SYSTEM = """\
You split a question about Washi (traditional Japanese paper) into the independent facts
that must be found to answer it.

Each sub-query is one thing to look up, phrased so it could be searched on its own.

Every sub-query MUST name at least one concrete thing from the question - a paper, a
workshop, a person, a place, a technique. A sub-query that names none of them, such as
"what broader category does that designation belong to", cannot be searched: there is
nothing to look up. Carry the relevant name into each sub-query even when the original
question only referred back to it with "that" or "it".

A question with a single constraint yields one sub-query - do not invent more. A question
combining two constraints ("made from X and used for Y") yields one per constraint,
because they are found in different parts of the knowledge and meet only at the answer.
A question that walks a chain ("A, then its region, then that region's country") yields
ONE sub-query naming the starting point: the walk itself will follow the rest, and
splitting it invents intermediate names that are not known yet.

Also list the concrete things the question NAMES - papers, workshops, people, places,
techniques, materials, institutions. Copy each as written in the question. These are the
places the search starts from, so completeness matters more than judgement: include a
name even if it looks incidental. Do NOT include the thing being asked *for* -- a good
question does not name its own answer, so anything you would have to guess is not a
mention.

Return ONLY JSON:
{"subqueries": ["...", "..."], "mentions": ["...", "..."]}   (at most 4 of each)
"""

DECOMPOSE_USER = """\
Question: {question}

Return the JSON.
"""

# ------------------------------------------------------------------------- navigation
NAVIGATE_SYSTEM = """\
You are exploring an encyclopedia of Washi (traditional Japanese paper) by following
links, looking for what answers a specific question.

You are standing on one page and make two separate decisions.

The page separates "## Facts" -- what this entity itself asserts -- from "## Mentioned by"
-- other entities that point at it. A question about this entity is normally answered from
its Facts; Mentioned by is for finding who produces, contains or designated it.

CITE - which links on this page are themselves part of the answer. A question asking what
is known about this subject is answered by the facts on this page, not by going elsewhere:
cite all of them. A question asking which thing satisfies two conditions is answered by
citing both conditions from the page of the thing that satisfies them. Cite nothing if
nothing here bears on the question.

Note that only cited links become evidence -- the prose at the top of a page informs your
decision but does not reach the reader. Do not skip citing a fact because the prose
already states it. Equally, do not cite a page wholesale: the evidence budget is small,
and irrelevant citations crowd out the facts that answer the question.

FOLLOW - which links to walk to next, because the answer lies further on. A question that
traces a route ("its town, then that town's prefecture") is answered by following, one
step at a time. Prefer a link that closes the gap over one that merely stays on topic;
being on a related page is not progress.

The two are independent. A page can be worth citing and not worth leaving, worth leaving
and not worth citing, or both.

Return ONLY JSON:
{"answered": true|false, "cite": ["<exact target>", ...], "follow": ["<exact target>", ...],
 "why": "<12 words>"}
- Copy target names exactly as written. "follow" may be empty; "cite" should be empty
  only when this page genuinely holds nothing relevant.
- "cite": at most {max_cite} targets. "follow": at most {max_choices} targets.
  Be generous when citing and sparing when following. If the question asks what is known
  about the subject of this page, cite EVERY link that states something about it - each
  omitted fact is a missing piece of the answer.
- "answered": true only when the OVERALL question is now fully settled by what you have
  cited plus the route already taken. Finding the first of several requested steps is not
  an answer. When in doubt, answer false and keep going.
"""

NAVIGATE_USER = """\
Overall question: {question}
Currently looking for: {subquery}

Path taken so far: {trail}

{page}

Reply with JSON only.
"""

# ----------------------------------------------------------------------- verification
VERIFY_SYSTEM = """\
You judge whether a chain of facts is worth keeping as evidence for one part of a
question.

Judge the chain as a whole, not fact by fact: individually true but unrelated facts
support nothing together.

**Judge it against the SUB-GOAL, not the whole question.** This chain was sent to find
one part of the answer, and other chains are finding the others; they are combined
afterwards. A question comparing two things is answered by two chains, one per thing, and
neither is complete on its own. Rejecting a chain for covering only its own half discards
evidence that was exactly right. Reject only what does not bear on the sub-goal at all.

Return ONLY JSON: {"verdict": "support" | "partial" | "reject", "why": "<15 words>"}
- support: this chain settles its sub-goal.
- partial: relevant to the sub-goal, but incomplete.
- reject: irrelevant to the sub-goal.
"""

VERIFY_USER = """\
Overall question: {question}
This chain's sub-goal: {subquery}

Chain of facts:
{chain}
{attrs}
Reply with JSON only.
"""

# ---------------------------------------------------------------------------- answering
ANSWER_SYSTEM = """\
You answer questions about Washi (traditional Japanese paper), grounding your answer in
the evidence provided - a set of facts retrieved from a knowledge graph, sometimes with
images.

**Always commit to an answer.** Lead with the most likely answer, in the first sentence,
even when the evidence is incomplete: if the evidence narrows the question to a few
candidates, name the best-supported one; if it settles the question, state it outright.
Describing what the evidence fails to show, and stopping there, is not an answer - a
reader learns nothing from it. Add a brief hedge afterwards if you must, but never let the
hedge replace the answer.

Prefer the evidence over your own recollection wherever they disagree, and do not invent
specifics the evidence does not support. Answer in at most 60 words.
"""

ANSWER_USER = """\
Question: {question}

Evidence:
{evidence}

Answer.
"""


# ---------------------------------------------------------------- ReAct baseline
# The generic agentic comparison. WikiWalk's gain over one-shot retrieval could be the
# agency rather than the design, and nothing in the main table separates the two: every
# baseline answers in a single call while WikiWalk takes about nineteen. This agent gets
# the same graph, the same backbone, the same evidence budget and a comparable number of
# calls, but none of WikiWalk's structure -- no page as the unit of observation, no
# separation of citing from following, no beam over partial chains, no chain-level
# verification, no sub-query decomposition. What it is missing is exactly what is being
# claimed.
REACT_SYSTEM = """\
You are answering a question about Washi (traditional Japanese paper) by exploring a
knowledge graph. You cannot see the graph; you must ask for parts of it.

Each turn, reply with one JSON object and nothing else:

  {"thought": "...", "action": "search", "arg": "<text>"}
      find entities whose names match the text
  {"thought": "...", "action": "expand", "arg": "<entity>", "keep": [1, 4]}
      list the facts attached to an entity. "keep" names facts to save as evidence by
      their numbers; omit it or use [] to save nothing.
  {"thought": "...", "action": "finish", "keep": [2]}
      stop exploring. Save any last facts with "keep".

Fact numbers are permanent. They keep counting up as you expand more entities, and a
number always refers to the same fact, so you can save a fact from an earlier observation
at any later turn.

Save a fact only if it helps answer the question. You have at most {steps} turns and may
save at most {budget} facts; saving more replaces nothing, the extras are dropped. Finish
as soon as you have what you need.
"""

# Offered only when the run enables --react-image-search and the item carries q_m: the
# same CLIP match over A_img that gives WikiWalk its image roots, exposed as a tool.
REACT_IMAGE_ACTION = """\
  {"thought": "...", "action": "image_search"}
      find the entities whose pictures look most like the image attached to the question
"""

REACT_USER = """\
Question: {question}

{history}
What is your next action?"""


# --------------------------------------------------------------- evidence selection
# Used only by the --model-select diagnostic. The default pipeline assembles the final
# evidence set by rule (round-robin over the ranked chains) and never asks the backbone
# which facts matter; this asks it, so that the two assembly strategies can be compared
# on the same walk.
SELECT_SYSTEM = """\
You choose which retrieved facts to hand to someone who must answer a question. Keep the
facts the answer actually needs and leave out the rest. A question that compares two
things needs facts about both of them.

Return ONLY JSON: {"keep": [<fact numbers>]}
"""

SELECT_USER = """\
Question: {question}

Facts found (choose at most {budget}):
{candidates}
"""
