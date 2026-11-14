# Free AI Model Aggregator

One server that speaks OpenAI Chat Completions and Anthropic Messages to the
client, and OpenAI, Anthropic and Gemini to ten simulated free-tier model
providers. An entry for the make.fun bounty
[free-model-aggregator-01](https://make.fun/bounty/free-model-aggregator-01)
("The best free AI model aggregator").

## What it does

- **Reads the brief** at `$MAKEFUN_PROMPT_FILE` (directory, quality table,
  workload) and **probes each provider's catalog route** in the provider's
  own wire, then builds one catalog: every provider that answered, every
  model it listed, models sorted by id and deduplicated.
- **Registers** the port and that catalog at `POST $MAKEFUN_ENV_URL/register`,
  and **polls** `GET $MAKEFUN_ENV_URL/work` every 100 ms, exiting 0 when the
  env says the workload is done.
- **Serves `POST /v1/chat/completions` and `POST /v1/messages`** (JSON and
  SSE), and **`GET /v1/models`** in the shape of the auth header the client
  sent (Bearer → OpenAI list, `x-api-key` → Anthropic list).
- **Routes each request to the best free model that meets its requirements**,
  by the brief's quality table (rating first, then provider id, then model
  id). Requirements come from the workload manifest row by arrival order,
  union the request body's own (tools, vision, JSON mode).
- **Fails over**: one attempt per provider per request; on a 402, 429 or 5xx
  it tries the next candidate. A provider whose budget is spent or that is
  out is never called again in the trial; a scheduled failure is transient
  and leaves the provider eligible for the next request.
- **Translates all six paths** between the client's wire and the provider's
  wire, passing the system prompt, user text (images included), tools, stop
  sequences, `max_tokens` and JSON mode through untouched — the providers
  echo what they parsed, and the client compares that echo with what it
  sent. Answers are buffered and rendered in the client's wire with the
  provider's real usage counts and stop reason, so an Anthropic stream's
  `message_start` carries the true input tokens and an OpenAI stream always
  carries its usage chunk (the upstream call forces
  `stream_options.include_usage`).

## Layout

```
main.ts            the server: brief, catalog, routes, register, poll
lib/brief.ts       brief parsing, auth headers, catalog probe, catalog build
lib/upstream.ts    provider calls in the three wires, one normalized answer
lib/translate.ts   request translation and response rendering, six paths
LICENSE            MIT
```

Standard library only — nothing is installed before the server runs.
Deterministic: no wall clock, no unseeded randomness, so a replay serves
the same bytes. `node main.ts --list` prints the catalog document byte for
byte as registered; outside a trial (no brief file) it prints
`{"providers":[]}`.

## Run

```sh
node main.ts            # serves, registers, polls /work, exits 0 when done
node main.ts --list     # prints the catalog document and stops
```

Requires Node 22+ (uses the built-in type stripping for `.ts`).
