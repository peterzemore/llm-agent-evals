# Case schema

One JSON object per line. Blank lines and `#` comments are ignored.

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | yes | Unique, stable. Referenced by results, labels, and gate output. |
| `utterance` | yes | What the caller said, as a transcriber would render it. |
| `category` | yes | Groups the per-category breakdown, e.g. `stock`, `adversarial`. |
| `expected_tool` | no | Tool name, or `null` to mean "answer directly, no tool call". |
| `expected_args` | no | Expected argument values, checked per `arg_rules`. |
| `arg_rules` | no | Per-argument match rule. Default `exact`. |
| `forbidden_phrases` | no | Substrings that must not appear. Zero tolerance. |
| `rubric` | no | Grading instruction for the LLM judge. Only cases with a rubric are judged. |
| `notes` | no | Why the case exists. For humans. |

`expected_tool: null` is a real expectation, not a missing value: answering
"what time do you close" from the system prompt instead of calling a tool saves
a round-trip of latency in the middle of a phone call.

## Match rules

| Rule | Behavior |
| --- | --- |
| `exact` | String equality. The default. |
| `iexact` | Case-insensitive, surrounding whitespace stripped. |
| `icontains` | Expected value appears somewhere in the actual value. |
| `numeric` | Both sides parse as numbers and compare equal. |
| `email` | Case-insensitive equality after stripping. |
| `empty` | Actual must be empty or absent - asserts the agent did **not** invent a value. |
| `regex:PAT` | `re.search(PAT, actual)`. |
| `any` | Anything, including nothing. |

An unrecognized rule raises rather than passing. A typo in a rule name must
never silently grade every case as correct.

Only arguments named in `expected_args` or `arg_rules` are checked. Extra
arguments are ignored, so adding an optional tool parameter does not
retroactively fail every stored case.

## Writing a good case

- **One behavior per case.** If it fails, the case id should tell you what
  broke without opening the file.
- **`empty` is underused.** The most common real failure in order lookups is
  the agent calling the tool with an invented or blank identifier rather than
  asking. `{"email": "empty"}` catches it.
- **Negative cases need a rubric, not just `expected_tool: null`.** Not calling
  a tool is necessary but not sufficient - the agent also has to say something
  useful.
- **Guard on leakage markers, not topic words.** See `docs/design.md`.
