// Provider calls: one normalized answer out of the three upstream wires.
//
// Every provider answers in its own wire's shape, but the server only ever
// needs four things from an answer: the text, the usage counts the provider
// reported, the stop family, and any tool calls it carried. The simulated
// providers never emit tool calls (their answers are one text and a stop),
// but the parsers below read them anyway, so the same code runs against the
// real services, where a model may call a tool.
//
// A call that is refused (401, 400, 402, 429, 5xx) is not an answer: it
// comes back as a Refusal carrying the status and the provider's own error
// message, so the router can tell a spent budget ("free-tier budget is
// spent", "is out") from a scheduled failure ("refused this call").
//
// Everything here is a pure function of the request and the provider's
// answer: no clock, no randomness.

import {
  authHeaders,
  isRecord,
  joinBase,
  type DirectoryProvider,
  type Json,
} from './brief.ts';

/** A tool call a provider's answer carried, in OpenAI's spelling. */
export interface ToolCall {
  id: string;
  name: string;
  /** The arguments as a JSON string, as the provider sent them. */
  arguments: string;
}

/** One provider answer, in the shape every renderer consumes. */
export interface UpstreamAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  /** Normalized stop family, in OpenAI's naming. */
  stop: 'stop' | 'length' | 'tool_calls' | 'content_filter';
  toolCalls: ToolCall[];
  model: string;
  providerId: string;
}

/** A refusal: the provider answered, but not with an answer. */
export class Refusal extends Error {
  readonly status: number;
  readonly code: string;
  constructor(status: number, code: string, message: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

const CALL_TIMEOUT_MS = 10000;
/** The most bytes of one provider answer this server reads. */
const MAX_ANSWER_BYTES = 2 * 1024 * 1024;

function numberOrZero(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/** The provider's error message out of its wire's error envelope. */
function errorMessage(wire: string, status: number, text: string): { code: string; message: string } {
  let code = '';
  let message = `the provider answered HTTP ${status}`;
  try {
    const body: unknown = JSON.parse(text);
    if (isRecord(body)) {
      const error = isRecord(body.error) ? body.error : null;
      if (error !== null) {
        if (typeof error.code === 'string') code = error.code;
        else if (typeof error.type === 'string') code = error.type;
        if (typeof error.message === 'string') message = error.message;
      }
    }
  } catch {
    // Not JSON: keep the HTTP status as the message.
  }
  if (wire === 'gemini' && code === '') code = String(status);
  return { code, message };
}

/** One SSE frame's lines: the event name (or null) and its data lines. */
interface SseFrame {
  event: string | null;
  data: string;
}

/** Split SSE text into frames, tolerating CRLF. */
export function parseSse(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const chunk of normalized.split('\n\n')) {
    if (chunk.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of chunk.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** Read a response body, capped, as text. */
async function readText(res: Response): Promise<string> {
  const buf = Buffer.from(await res.arrayBuffer());
  return buf.subarray(0, MAX_ANSWER_BYTES).toString('utf8');
}

/** The text of a string or of a block array's text blocks. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
}

/** A tool call's arguments as a JSON string. */
function argumentsText(value: unknown): string {
  if (typeof value === 'string') return value;
  if (isRecord(value)) return JSON.stringify(value);
  return '{}';
}

/** OpenAI's finish_reason, normalized. */
function openAIStop(finish: unknown): UpstreamAnswer['stop'] {
  if (finish === 'length') return 'length';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_calls';
  if (finish === 'content_filter') return 'content_filter';
  return 'stop';
}

/** Anthropic's stop_reason, normalized. */
function anthropicStop(reason: unknown): UpstreamAnswer['stop'] {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason, normalized. */
function geminiStop(reason: unknown): UpstreamAnswer['stop'] {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** An OpenAI Chat Completions JSON body as a normalized answer. */
function openAIJsonAnswer(body: Json, model: string, providerId: string): UpstreamAnswer | null {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const toolCalls: ToolCall[] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      toolCalls.push({
        id: typeof call.id === 'string' ? call.id : '',
        name: typeof call.function.name === 'string' ? call.function.name : '',
        arguments: argumentsText(call.function.arguments),
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    inputTokens: numberOrZero(usage.prompt_tokens),
    outputTokens: numberOrZero(usage.completion_tokens),
    stop: openAIStop(choice.finish_reason),
    toolCalls,
    model,
    providerId,
  };
}

/** An OpenAI SSE stream as a normalized answer. */
function openAIStreamAnswer(text: string, model: string, providerId: string): UpstreamAnswer | null {
  let out = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop: UpstreamAnswer['stop'] = 'stop';
  let sawAny = false;
  const toolCalls: ToolCall[] = [];
  for (const frame of parseSse(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    sawAny = true;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    if (Array.isArray(delta.tool_calls)) {
      for (const call of delta.tool_calls) {
        if (!isRecord(call)) continue;
        const index = typeof call.index === 'number' ? call.index : toolCalls.length;
        const fn = isRecord(call.function) ? call.function : {};
        const existing = toolCalls[index];
        if (existing === undefined) {
          toolCalls[index] = {
            id: typeof call.id === 'string' ? call.id : '',
            name: typeof fn.name === 'string' ? fn.name : '',
            arguments: typeof fn.arguments === 'string' ? fn.arguments : '',
          };
        } else {
          if (typeof fn.arguments === 'string') existing.arguments += fn.arguments;
          if (existing.name === '' && typeof fn.name === 'string') existing.name = fn.name;
          if (existing.id === '' && typeof call.id === 'string') existing.id = call.id;
        }
      }
    }
    if (typeof choice.finish_reason === 'string') stop = openAIStop(choice.finish_reason);
  }
  if (!sawAny) return null;
  return { text: out, inputTokens, outputTokens, stop, toolCalls, model, providerId };
}

/** An Anthropic Messages JSON body as a normalized answer. */
function anthropicJsonAnswer(body: Json, model: string, providerId: string): UpstreamAnswer | null {
  if (body.type !== 'message' || !Array.isArray(body.content)) return null;
  const text = textOf(body.content);
  const toolCalls: ToolCall[] = [];
  for (const block of body.content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    toolCalls.push({
      id: typeof block.id === 'string' ? block.id : '',
      name: typeof block.name === 'string' ? block.name : '',
      arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    inputTokens: numberOrZero(usage.input_tokens),
    outputTokens: numberOrZero(usage.output_tokens),
    stop: anthropicStop(body.stop_reason),
    toolCalls,
    model,
    providerId,
  };
}

/** An Anthropic SSE stream as a normalized answer. */
function anthropicStreamAnswer(text: string, model: string, providerId: string): UpstreamAnswer | null {
  let out = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop: UpstreamAnswer['stop'] = 'stop';
  let sawAny = false;
  const toolCalls: ToolCall[] = [];
  const toolByIndex = new Map<number, ToolCall>();
  for (const frame of parseSse(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    sawAny = true;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = numberOrZero(data.message.usage.input_tokens);
    }
    if (frame.event === 'content_block_start' && isRecord(data.content_block)) {
      const block = data.content_block;
      if (block.type === 'tool_use' && typeof data.index === 'number') {
        const call: ToolCall = {
          id: typeof block.id === 'string' ? block.id : '',
          name: typeof block.name === 'string' ? block.name : '',
          arguments: '',
        };
        toolByIndex.set(data.index, call);
        toolCalls.push(call);
      }
    }
    if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      const delta = data.delta;
      if (delta.type === 'text_delta' && typeof delta.text === 'string') out += delta.text;
      if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && typeof data.index === 'number') {
        const call = toolByIndex.get(data.index);
        if (call !== undefined) call.arguments += delta.partial_json;
      }
    }
    if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = anthropicStop(data.delta.stop_reason);
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    }
  }
  if (!sawAny) return null;
  return { text: out, inputTokens, outputTokens, stop, toolCalls, model, providerId };
}

/** One Gemini candidate's text: its parts' text joined. */
function geminiText(candidate: Json): string {
  const content = isRecord(candidate.content) ? candidate.content : null;
  if (content === null || !Array.isArray(content.parts)) return '';
  return content.parts
    .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
    .join('');
}

/** A Gemini generateContent JSON body as a normalized answer. */
function geminiJsonAnswer(body: Json, model: string, providerId: string): UpstreamAnswer | null {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null) return null;
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: geminiText(candidate),
    inputTokens: numberOrZero(usage.promptTokenCount),
    outputTokens: numberOrZero(usage.candidatesTokenCount),
    stop: geminiStop(candidate.finishReason),
    toolCalls: [],
    model,
    providerId,
  };
}

/** A Gemini streamGenerateContent SSE stream as a normalized answer. */
function geminiStreamAnswer(text: string, model: string, providerId: string): UpstreamAnswer | null {
  let out = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop: UpstreamAnswer['stop'] = 'stop';
  let sawAny = false;
  for (const frame of parseSse(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    sawAny = true;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      out += geminiText(candidate);
      if (typeof candidate.finishReason === 'string') stop = geminiStop(candidate.finishReason);
    }
    if (isRecord(data.usageMetadata)) {
      inputTokens = numberOrZero(data.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
  }
  if (!sawAny) return null;
  return { text: out, inputTokens, outputTokens, stop, toolCalls: [], model, providerId };
}

export interface UpstreamCall {
  provider: DirectoryProvider;
  model: string;
  /** The translated request body in the provider's own wire. */
  body: Json;
  stream: boolean;
}

/**
 * One call to a provider, in its own wire. Returns the normalized answer on
 * a 2xx, throws Refusal on any other status, and throws on a transport
 * failure or an answer that does not parse.
 */
export async function callProvider(envUrl: string, call: UpstreamCall): Promise<UpstreamAnswer> {
  const { provider, model } = call;
  const base = joinBase(envUrl, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
  } else {
    const encoded = encodeURIComponent(model);
    url = call.stream
      ? `${base}/v1beta/models/${encoded}:streamGenerateContent?alt=sse`
      : `${base}/v1beta/models/${encoded}:generateContent`;
  }
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(call.body),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch (err) {
    throw new Error(`calling ${provider.id} failed: ${(err as Error).message}`);
  }
  const text = await readText(res);
  if (!res.ok) {
    const { code, message } = errorMessage(provider.wire, res.status, text);
    throw new Refusal(res.status, code, message);
  }
  let answer: UpstreamAnswer | null = null;
  if (provider.wire === 'openai') {
    if (call.stream) {
      answer = openAIStreamAnswer(text, model, provider.id);
    } else {
      let parsed: unknown;
      try {
        parsed = JSON.parse(text);
      } catch {
        parsed = null;
      }
      answer = isRecord(parsed) ? openAIJsonAnswer(parsed, model, provider.id) : null;
    }
  } else if (provider.wire === 'anthropic') {
    if (call.stream) {
      answer = anthropicStreamAnswer(text, model, provider.id);
    } else {
      let parsed: unknown;
      try {
        parsed = JSON.parse(text);
      } catch {
        parsed = null;
      }
      answer = isRecord(parsed) ? anthropicJsonAnswer(parsed, model, provider.id) : null;
    }
  } else {
    if (call.stream) {
      answer = geminiStreamAnswer(text, model, provider.id);
    } else {
      let parsed: unknown;
      try {
        parsed = JSON.parse(text);
      } catch {
        parsed = null;
      }
      answer = isRecord(parsed) ? geminiJsonAnswer(parsed, model, provider.id) : null;
    }
  }
  if (answer === null) throw new Error(`${provider.id} answered ${provider.wire} in a shape this server cannot read`);
  return answer;
}
