# Two phone calls, four bugs, zero exceptions
The agent that answers my store's phone has never crashed. That is the problem.

It is a voice agent with four tools: check stock, look up an order, check loyalty
points, log a callback request. When it fails, it fails in complete sentences. It
reaches for the order tool to answer a question about opening hours. It passes a
caller's whole sentence into the product search. It agrees with a caller who insists
their order shipped. None of that shows up in an error rate. A customer notices first.

So I built an evaluation harness for it, and this post is about the part of that work
that actually changed the agent, which was not the harness. It was the two phone calls
I made after the harness existed.

## What the harness does

It scores the agent against a labeled set of cases and blocks a merge when a prompt
change moves the numbers the wrong way. Today that is 38 cases, 92.1% task success,
97.4% tool accuracy. Predictions are recorded once and replayed, so the gate runs on
every pull request with no API key and no spend, and with model behavior frozen, any
metric movement is attributable to a scoring change alone.

Thirty-eight cases supports the headline to roughly the nearest three points. The
per-category rows hold three to six cases each and are directional only. The honest
read of the results table is the order-lookup row at 60%, not the headline.

## The two calls

With the suite green, I called the agent from my own phone and listened like a
customer. Two calls surfaced four bugs that the suite had no case for.

**The callback nobody could make.** I asked for a call back and said my number. The
agent confirmed it, logged it, and told me someone would be in touch. The log entry
looked fine. It held the last seven digits. Speech-to-text had dropped the area code
and the agent stored what it heard. Every mechanical check passed: right tool, right
argument shape, polite confirmation. The caller would simply never hear from anyone.

**A policy it was never given.** I asked whether the store hosts birthday parties. It
said "no, we don't." That fact appears nowhere in anything the agent was given. A
denial is a policy claim exactly as much as an affirmation is, and this one was
invented.

**Two versions of the website.** The greeting said the store's web address correctly.
A stock referral later in the same call said it differently, because tool results are
relayed close to verbatim and one of them carried a bare domain. The hybrid it produced
names a domain the store does not own.

**The full inventory list.** Asked a broad stock question, it read out everything it
found.

All four pass the checks a matcher can do. They fail on what gets said after the tool
returns. That is the class of bug a deterministic assertion cannot see, and it is the
reason the suite has a second half.

## What changed in the harness

Each bug became a case before the fix was allowed to count. That order matters. A fix
without a pinned case is a fix that can quietly regress on the next prompt edit.

Two mechanics came out of it. First, cases can prime the conversation with a tool result
and grade the second turn, because that is where these failures live. Second, the
cheapest half of the suite caught all four: guarded phrases, meaning strings that must
not appear in the response. No judge required.

There is also a lesson from an earlier case that generalizes. A caller says "my son's
birthday is Saturday and he really likes Spiderman," and the whole sentence goes into the
product search. The assertion that catches it is a negative lookahead on the argument,
not a substring check, because "contains spiderman" is true of the broken input too. An
assertion that the right thing is present usually needs a matching assertion that the
wrong thing is absent.

## The judge, and the number I will not quote

Some cases are about wording, such as declining without being rude, and those carry a
rubric graded by a model. Three rules keep that judge honest. It can only veto: a judge
that likes a response cannot rescue a wrong tool call. It never sees the answer key. And
it is calibrated against blind human labels using Cohen's kappa rather than raw
agreement, because on a suite where 90% of cases pass, a judge that says "pass" to
everything scores 90% agreement and looks excellent. Kappa scores it zero. There is a
unit test asserting exactly that.

The current calibration is kappa 1.00 over 19 labels. I am not quoting that as evidence,
and the tool refuses to quote it either: it prints a warning that kappa is not meaningful
below 30 labels. The set is small and skewed 18 to 1 toward pass, so a single
disagreement would swing the estimate by roughly 0.4. Perfect agreement on a small,
skewed set is what an easy task looks like, not what a good judge looks like. The honest
reading is that the judge and the human have not yet disagreed anywhere, which is a
prerequisite for trusting it, not proof of it.

## What I would tell someone starting this

Build the deterministic checks first; they carry the suite. Scope the judge and let it
veto only. Report kappa, not agreement, and report the denominator next to every number.
Then put the harness down, call your own agent, and listen. The suite tells you when you
have broken something you already knew about. The phone call tells you what you did not
know to test.

Repo: github.com/peterzemore/llm-agent-evals
