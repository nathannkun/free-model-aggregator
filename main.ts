// The best free AI model aggregator.
//
// One server that speaks OpenAI Chat Completions and Anthropic Messages to
// the client, and OpenAI, Anthropic and Gemini to ten simulated free-tier
// providers. Each request goes to the best free model that meets its
// requirements, by the brief's quality table; when a provider refuses, the
// request fails over to the next candidate, one attempt per provider, until
// one answers. A provider whose budget is spent or that is out is never
// called again in the trial. The catalog it read is the document it
// registers and that `node main.ts --list` prints, byte for byte.
//
// The server is a pure function of the brief and the providers' answers:
// no wall clock, no unseeded randomness, so a replay serves the same bytes.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  isFreeModel,
  isRecord,
  meetsRequirements,
  probeProvider,
  ratingRowOf,
  readBrief,
  requirementsFromBody,
  type Catalog,
  type DirectoryProvider,
  type Json,
  type ListedModel,
  type Probe,
  type Requirement,
} from './lib/brief.ts';
import { callProvider, Refusal, type UpstreamAnswer } from './lib/upstream.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToOpenAI,
  clientToGemini,
  openAIError,
  openAIModelList,
  openAIRequest,
  openAIToAnthropic,
  renderAnthropicJson,
  renderAnthropicStream,
  renderOpenAIJson,
  renderOpenAIStream,
} from './lib/translate.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Providers whose budget is spent or that are out, for the rest of the trial. */
const exhausted = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
}

const byId = (a: { id: string }, b: { id: string }): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, deduplicated, sorted by id: what GET /v1/models lists. */
function servedModels(): { id: string; provider: string; display_name: string }[] {
  const seen = new Set<string>();
  const out: { id: string; provider: string; display_name: string }[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingRowOf(brief.ratings, probe.provider.id, model.id);
      if (!isFreeModel(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out.sort(byId);
}

interface Candidate {
  provider: DirectoryProvider;
  model: ListedModel;
  /** The quality table's rating_milli, or 0 where the table has no row. */
  rating: number;
}

/**
 * Every free model that meets one request's requirements, best first: the
 * quality table's rating, then the provider id, then the model id. The order
 * is static — it never depends on how many calls a provider has served, so
 * the routing is deterministic across a replay.
 */
function candidatesFor(requires: Requirement): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (exhausted.has(probe.provider.id)) continue;
    for (const model of probe.models) {
      const row = ratingRowOf(brief.ratings, probe.provider.id, model.id);
      if (!isFreeModel(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ provider: probe.provider, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, union the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.manifest[manifestIndex];
  manifestIndex += 1;
  const fromBody = requirementsFromBody(body);
  if (entry === undefined) return fromBody;
  return {
    tools: entry.requires.tools || fromBody.tools,
    vision: entry.requires.vision || fromBody.vision,
    json: entry.requires.json || fromBody.json,
    min_context: entry.requires.min_context,
  };
}

/** The request body one candidate's provider expects, in its own wire. */
function upstreamBody(api: 'openai' | 'anthropic', body: Json, candidate: Candidate): Json {
  const { provider, model } = candidate;
  if (provider.wire === 'openai') {
    return api === 'openai' ? openAIRequest(body, model.id) : anthropicToOpenAI(body, model.id);
  }
  if (provider.wire === 'anthropic') {
    return api === 'openai' ? openAIToAnthropic(body, model.id) : { ...body, model: model.id };
  }
  return clientToGemini(body, api);
}

interface ServeResult {
  ok: boolean;
  status: number;
  text: string;
  contentType: string;
}

/**
 * Serve one client request: try the candidates in order, one attempt per
 * provider, and answer from the first that answers. A refusal is the
 * provider's own error message, and the message alone tells a spent budget
 * ("free-tier budget is spent") or an outage ("is out") from a scheduled
 * failure ("refused this call"): the first two mark the provider exhausted
 * for the rest of the trial, the third leaves it eligible for the next
 * request.
 */
async function serve(api: 'openai' | 'anthropic', body: Json): Promise<ServeResult> {
  const stream = body.stream === true;
  const requires = requirementsFor(body);
  const candidates = candidatesFor(requires);
  const tried = new Set<string>();
  let lastError: Error | null = null;
  for (const candidate of candidates) {
    if (tried.has(candidate.provider.id)) continue;
    tried.add(candidate.provider.id);
    let answer: UpstreamAnswer;
    try {
      answer = await callProvider(ENV_URL, {
        provider: candidate.provider,
        model: candidate.model.id,
        body: upstreamBody(api, body, candidate),
        stream,
      });
    } catch (err) {
      if (err instanceof Refusal) {
        if (err.message.includes('free-tier budget is spent') || err.message.endsWith(' is out')) {
          exhausted.add(candidate.provider.id);
        }
        lastError = err;
        continue;
      }
      lastError = err as Error;
      continue;
    }
    const text =
      api === 'openai'
        ? stream
          ? renderOpenAIStream(answer)
          : JSON.stringify(renderOpenAIJson(answer))
        : stream
          ? renderAnthropicStream(answer)
          : JSON.stringify(renderAnthropicJson(answer));
    return { ok: true, status: 200, text, contentType: stream ? 'text/event-stream' : 'application/json' };
  }
  const message = lastError === null ? 'no free model meets this request' : lastError.message;
  return { ok: false, status: 503, text: message, contentType: 'application/json' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  sendText(res, status, 'application/json', JSON.stringify(body));
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const result = await serve('openai', body);
  if (result.ok) {
    sendText(res, 200, result.contentType, result.text);
    return;
  }
  sendError(res, 'openai', result.status, 'server_error', result.text);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const result = await serve('anthropic', body);
  if (result.ok) {
    sendText(res, 200, result.contentType, result.text);
    return;
  }
  sendError(res, 'anthropic', result.status, 'overloaded_error', result.text);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/**
 * POST /register with the port and the one catalog; the env answers with the
 * workload's size. Retried, because a register lost to a slow env would end
 * the trial before it starts.
 */
async function register(port: number, report: Catalog): Promise<void> {
  let last: unknown = null;
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      last = err;
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
  }
  throw last instanceof Error ? last : new Error(String(last));
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent)
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
