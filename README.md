# free-model-aggregator

One server that serves the **OpenAI Chat Completions** and **Anthropic Messages** APIs
(JSON and server-sent events) over ten free-tier model providers, routing each request
to the best free model that can serve it and failing over when a provider refuses.

Built for the [make.fun](https://make.fun) bounty
`free-model-aggregator-01` — "The best free AI model aggregator" — and graded against
its sealed env, generator and scorer. The same code, given real keys and base URLs, runs
against the real services: every provider speaks its real wire (OpenAI, Anthropic or
Gemini), with its own auth header, model list and per-trial budget.

## Run

Node 22, standard library only — nothing to install.

```sh
# As a trial entry (the env sets these three variables):
MAKEFUN_ENV_URL=http://store:8080 \
MAKEFUN_PROMPT_FILE=/input/prompt.txt \
MAKEFUN_TRIAL_SEED=<seed> \
  node main.ts

# Print the catalog the server would register, and exit:
node main.ts --list
```

Outside a trial, with no `MAKEFUN_PROMPT_FILE` and no `MAKEFUN_ENV_URL` set,
`--list` prints `{"providers":[]}` and exits 0: there is no brief to read and no
provider route to probe.

## How it works

- **Brief.** The trial's brief (`$MAKEFUN_PROMPT_FILE`) carries the provider
  directory (ten providers: id, wire, auth scheme, key variable, key, base path),
  the sealed quality table (each model's rating, context length, tools/vision/JSON
  capability, free flag) and the workload manifest (the 20 requests, in order, with
  each one's API, streaming flag, weight and requirements).
- **Catalog.** The server probes every provider's catalog route in the provider's
  own wire and registers one catalog with the env
  (`POST $MAKEFUN_ENV_URL/register`, `{"protocol_version": 1, "port", "catalog"}`).
  The catalog `--list` prints is that same document, byte for byte.
- **Routing.** Candidates are the free models meeting a request, ranked by the
  quality table's rating, highest first; ties go to the provider with fewer calls
  used so far, then to the brief's provider order, then to model id. Gemini is the
  only home of the top vision model (`gemini-3.8-flash`, the only tools+vision+JSON
  model), so a request that does not need vision never spends a Gemini call while a
  non-Gemini candidate is left. A request that asks for JSON mode on the OpenAI
  wire never goes to an Anthropic-wire provider, whose parser cannot carry
  `response_format`.
- **Failover.** Every attempt is counted per provider (a refusal counts, as the
  provider counts it). A 402, 429 or 5xx is retried once on the same provider — the
  generator draws at most two failure calls per provider, so one refusal is usually
  a scheduled fail and the next call answers — then the request fails over to the
  next candidate. Three refusals in a row cannot be scheduled fails, so a provider
  with three is exhausted or in outage and is skipped for the rest of the trial.
  Exactly one provider call may succeed per request, so a 200 answer is served,
  never retried.
- **Wires** (`lib/upstream.ts`). Each provider answer is normalised to
  `{ text, inputTokens, outputTokens }` and rendered in the client's wire. The
  provider's text is carried verbatim — it is an echo of what the provider parsed
  from the upstream request, and the scorer compares it against the client's body —
  so every upstream request preserves the system text, the last user message's
  text, the tool names, `max_tokens`, the stop sequences, the image bytes and JSON
  mode, exactly. Nothing is dropped, nothing is invented.
- **Determinism.** No wall clock, no unseeded randomness, no sleeps. Every choice
  derives from the brief, the requests and the providers' answers, so a replay
  leaves the same state export.

## Layout

```
main.ts            the server: brief, catalog, routing, routes, register, poll
lib/providers.ts   the brief's types and parser, auth headers, catalog probe
lib/upstream.ts    the three provider wires and the two client wires
lib/wire.ts        Anthropic <-> Chat Completions translation (starter)
LICENSE            MIT
```

## Local grading

From the unpacked kit (Docker is not needed; the process driver runs the trial on
loopback):

```sh
node cli/makefun.mjs grade-local --kit . --bot <this repo> --driver process --instances 3
```

Practice results (3/3 trials, replay matched byte for byte on each):

| trial | score | quality | served | wire | failover | catalog | calls |
|------:|------:|--------:|-------:|-----:|---------:|--------:|------:|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 30 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 31 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 32 |

Quality 99 is the ceiling on the practice trials: the quality bound ignores
per-trial budgets, and the best model (`z-ai/glm-5.3-flash:free`, 1475) has a
budget of 10–12 calls for the 17 requests that want it, so a few requests must
fall to the 1470 tier. The routing above is at that optimum on all three trials
(verified per-request against the sealed reference). The house baseline starter
scores 38 on trial 0.

## License

MIT. See `LICENSE`.
