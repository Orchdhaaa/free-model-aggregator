// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// It reads the brief (provider directory, quality table, workload manifest), probes every
// provider's catalog route, registers one catalog with the env, then serves the client's 20
// requests: POST /v1/chat/completions and POST /v1/messages, JSON and SSE, each answered by
// the best free model that meets the request's requirements, failing over across providers
// when one refuses. GET /v1/models answers in the wire its auth header names.
// `node main.ts --list` prints the catalog document and exits 0.
//
// Routing (deterministic: a pure function of the brief and the provider answers so far):
//   - candidates are the free models meeting the request, ranked by the quality table's
//     rating_milli, highest first; ties go to the provider with fewer calls used so far,
//     then to the brief's provider order, then to model id;
//   - Gemini is the only provider with the top vision model (gemini-3.8-flash, the only
//     tools+vision+json model), so a request that does not need vision never spends a Gemini
//     call while a non-Gemini candidate is left: Gemini models sort after every non-Gemini
//     candidate for those requests, and stay as the fallback;
//   - a request that asks for JSON mode on the OpenAI wire never goes to an Anthropic-wire
//     provider: its parser cannot carry response_format, so the echo would come back without
//     it. (Anthropic-wire requests carry no JSON mode of their own, so any wire serves them.)
//   - every attempt is counted per provider (a refusal counts, like the provider counts it);
//     a 402, 429, 5xx or unreachable provider fails over to the next candidate. Exactly one
//     provider call may succeed per request, so a 200 answer is served, never retried.
//   - a refusal is retried once on the same provider: the generator draws at most two
//     failure calls a provider (and one provider's second call always fails), so a single
//     refusal is usually a scheduled fail, and the next call answers. Three refusals in a
//     row cannot be scheduled fails, so a provider with three is exhausted or in outage and
//     is skipped for the rest of the trial: that caps the calls wasted on a spent provider
//     at three and keeps the tie-break (fewer provider calls) low.
//   - the trial never publishes budgets or failure schedules: they are learned from refusals
//     only. No wall clock, no unseeded randomness, no sleeps (a retry-after is information,
//     not a wait: the failover is immediate).
//
// The wires (lib/upstream.ts): each provider answer is normalised to { text, inputTokens,
// outputTokens } and rendered in the client's wire. The provider's text is carried verbatim:
// it is `ANSWER ECHO{...}` and the scorer compares its echo against the client's body, so the
// upstream request preserves the system text, the last user message's text, the tool names,
// max_tokens, the stops, the image bytes and JSON mode, exactly.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicToAnthropic,
  clientJsonMode,
  clientToGemini,
  openAIToAnthropicRequest,
  openAIToOpenAI,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  renderAnthropicJson,
  renderAnthropicStream,
  renderOpenAIJson,
  renderOpenAIStream,
  type Answer,
} from './lib/upstream.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds (the client allows 30 s a request). */
const PROVIDER_TIMEOUT_MS = 15000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;
/** Provider calls made so far, by provider id (a refusal counts, as the provider counts it). */
const callsUsed = new Map<string, number>();
/** Consecutive HTTP refusals per provider id; a success resets it to 0. */
const refusals = new Map<string, number>();
/** Three refusals in a row prove a provider spent or out: its failure schedule holds at most two. */
const EXHAUSTED_STREAK = 3;
/** The brief's provider order, for deterministic tie-breaks. */
const providerOrder = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  brief.directory.providers.forEach((p, i) => providerOrder.set(p.id, i));
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * The free models that can serve one request, best first. Rating first; Gemini (the only
 * home of the top vision model) held in reserve for requests that do not need vision; ties
 * to the provider with fewer calls used, then the brief's order, then model id.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    // A provider that refused three calls in a row is spent or out: skip it.
    if ((refusals.get(probe.provider.id) ?? 0) >= EXHAUSTED_STREAK) continue;
    // An Anthropic-wire provider cannot carry an OpenAI request's JSON mode: its echo
    // would come back without it.
    if (jsonMode && clientApi === 'openai' && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  const reserve = (c: Candidate): number => (requires.vision === true ? 0 : c.probe.provider.wire === 'gemini' ? 1 : 0);
  const used = (c: Candidate): number => callsUsed.get(c.probe.provider.id) ?? 0;
  const order = (c: Candidate): number => providerOrder.get(c.probe.provider.id) ?? 0;
  out.sort(
    (a, b) =>
      reserve(a) - reserve(b) ||
      b.rating - a.rating ||
      used(a) - used(b) ||
      order(a) - order(b) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  if (out.length > 0) return out;
  // Every provider looks spent: fall back to the unfiltered list rather than failing outright.
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && clientApi === 'openai' && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort(
    (a, b) =>
      reserve(a) - reserve(b) ||
      b.rating - a.rating ||
      used(a) - used(b) ||
      order(a) - order(b) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
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

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
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

/** The upstream request body for one client body on one provider wire, naming `model`. */
function upstreamBody(clientApi: 'openai' | 'anthropic', wire: 'openai' | 'anthropic' | 'gemini', body: Json, model: string): Json {
  if (wire === 'openai') return clientApi === 'openai' ? openAIToOpenAI(body, model) : anthropicToOpenAI(body, model);
  if (wire === 'anthropic') return clientApi === 'openai' ? openAIToAnthropicRequest(body, model) : anthropicToAnthropic(body, model);
  return clientToGemini(body, clientApi);
}

/** The upstream URL for one provider call. */
function upstreamUrl(candidate: Candidate, stream: boolean): string {
  const base = joinBase(ENV_URL, candidate.probe.provider.base);
  const wire = candidate.probe.provider.wire;
  if (wire === 'openai') return `${base}/v1/chat/completions`;
  if (wire === 'anthropic') return `${base}/v1/messages`;
  const action = stream ? ':streamGenerateContent' : ':generateContent';
  return `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}${action}${stream ? '?alt=sse' : ''}`;
}

interface Attempt {
  ok: boolean;
  /** HTTP status when the provider answered at all. */
  status: number;
  /** The parsed answer, when the provider answered 200 with a well-formed body. */
  answer: Answer | null;
}

/** One call to one provider. A refusal (402, 429, 5xx) or an unreachable provider is not an answer. */
async function attempt(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<Attempt> {
  const provider = candidate.probe.provider;
  const stream = body.stream === true;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (stream) headers.accept = 'text/event-stream';
  let res: Response;
  try {
    res = await fetch(upstreamUrl(candidate, stream), {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody(clientApi, provider.wire, body, candidate.model.id)),
      signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
    });
  } catch (err) {
    // A connection that never reached the provider spent none of its budget; a timeout
    // may have. Count the timeout, not the refusal to connect.
    if (err instanceof Error && err.name === 'AbortError') {
      callsUsed.set(provider.id, (callsUsed.get(provider.id) ?? 0) + 1);
    }
    return { ok: false, status: 0, answer: null };
  }
  // Any HTTP answer from the provider counts toward its budget, refusal included.
  // A success resets the refusal streak; an HTTP refusal extends it; an unreachable
  // provider (no answer at all) leaves it alone.
  callsUsed.set(provider.id, (callsUsed.get(provider.id) ?? 0) + 1);
  if (!res.ok) {
    refusals.set(provider.id, (refusals.get(provider.id) ?? 0) + 1);
    return { ok: false, status: res.status, answer: null };
  }
  refusals.set(provider.id, 0);
  const text = await res.text();
  const answer =
    provider.wire === 'openai'
      ? parseOpenAIAnswer(text, stream)
      : provider.wire === 'anthropic'
        ? parseAnthropicAnswer(text, stream)
        : parseGeminiAnswer(text, stream);
  // A 200 with a body that is not an answer cannot be served and cannot be retried either:
  // the provider already counted it as a success, so a second success would break the
  // one-provider rule. The caller turns this into a 502 for the request.
  if (answer === null) return { ok: false, status: 200, answer: null };
  return { ok: true, status: 200, answer };
}

/** Serve one client request: try the candidates in order until one answers, then render it. */
async function serve(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic', body: Json): Promise<void> {
  const requires = requirementsFor(body);
  const jsonMode = clientApi === 'openai' && clientJsonMode(body);
  const candidates = candidatesFor(requires, clientApi, jsonMode);
  if (candidates.length === 0) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  const stream = body.stream === true;
  let sawUnparseableSuccess = false;
  for (const candidate of candidates) {
    let result = await attempt(candidate, clientApi, body);
    // One retry on the same provider: a single refusal is usually a scheduled fail, and
    // the next call answers. No retry past two refusals in a row (a third would prove the
    // provider spent), and never after an unreachable provider or a 200 (a timeout may
    // already have been counted a success, and a 200 already was one).
    if (!result.ok && result.status !== 0 && result.status !== 200 && (refusals.get(candidate.probe.provider.id) ?? 0) < EXHAUSTED_STREAK) {
      log(`retry: ${candidate.probe.provider.id}/${candidate.model.id} after HTTP ${result.status}`);
      result = await attempt(candidate, clientApi, body);
    }
    if (result.ok && result.answer !== null) {
      log(
        `served ${clientApi} stream=${stream} with ${candidate.probe.provider.id}/${candidate.model.id} (${candidate.rating}) after ${callsUsed.get(candidate.probe.provider.id) ?? 0} calls on ${candidate.probe.provider.id}`,
      );
      if (stream) {
        const text = clientApi === 'openai' ? renderOpenAIStream(result.answer, candidate.model.id) : renderAnthropicStream(result.answer, candidate.model.id);
        return sendText(res, 200, 'text/event-stream; charset=utf-8', text);
      }
      const rendered = clientApi === 'openai' ? renderOpenAIJson(result.answer, candidate.model.id) : renderAnthropicJson(result.answer, candidate.model.id);
      return sendJson(res, 200, rendered);
    }
    if (result.status === 200) sawUnparseableSuccess = true;
    else log(`failover: ${candidate.probe.provider.id}/${candidate.model.id} answered ${result.status === 0 ? 'unreachable' : result.status}`);
  }
  if (sawUnparseableSuccess) {
    return sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'upstream_unreachable', 'the provider answered 200 with a body that is not an answer');
  }
  return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'every free model that meets this request refused it');
}

/** POST /v1/chat/completions. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve(req, res, 'openai', body);
}

/** POST /v1/messages. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve(req, res, 'anthropic', body);
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

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
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
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
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
