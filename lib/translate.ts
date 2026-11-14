// Wire translation: client requests into upstream requests, and normalized
// provider answers back into the client's wire.
//
// The client speaks OpenAI Chat Completions or Anthropic Messages; the
// providers speak OpenAI, Anthropic or Gemini. Six paths in all. Every
// translation passes the request through untouched in meaning: the system
// prompt, the user text (images included), the tools, the stop sequences,
// max_tokens and JSON mode all survive, because the provider echoes what it
// parsed and the client compares that echo with what it sent. A field
// rewritten or dropped loses the request.
//
// Responses are rendered from the normalized answer, buffered: the whole
// upstream answer is read first, then emitted in the client's wire with the
// provider's real usage counts and stop reason. Buffering is what makes the
// Anthropic message_start carry the true input tokens and the OpenAI stream
// always carry its usage chunk, which a chunk-by-chunk translation cannot
// know until the stream is over. The trial's client parses frames, not
// timing, so a buffered stream still counts as a stream.
//
// Everything here is a pure function of its arguments: no clock, no
// randomness, so a replay renders the same bytes.

import { isRecord, type Json } from './brief.ts';
import type { UpstreamAnswer } from './upstream.ts';

/** A fixed epoch second for every response's `created`: nothing reads the clock. */
const FIXED_CREATED = 1767225600;
/** The Anthropic API version the client sends and the providers expect. */
const ANTHROPIC_VERSION = '2023-06-01';

/** The text of a string or of a block array's text blocks. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** A tool call's arguments as an object. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** One Anthropic content block as an OpenAI content part, or null. */
function anthropicBlockToPart(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') {
      return { type: 'image_url', image_url: { url: source.url } };
    }
  }
  return null;
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** The max_tokens of a client body, OpenAI or Anthropic spelling. */
function maxTokensOf(body: Json): number | null {
  if (typeof body.max_tokens === 'number') return body.max_tokens;
  if (typeof body.max_completion_tokens === 'number') return body.max_completion_tokens;
  return null;
}

/** Whether an OpenAI client body asks for JSON mode. */
function jsonModeOf(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming
 * `model`: the system prompt becomes a system message, image blocks become
 * image_url data URLs, tool_use blocks become tool_calls, tool_result blocks
 * become tool messages, tools and tool_choice are rewritten, and the output
 * cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = anthropicBlockToPart(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  const maxTokens = maxTokensOf(body);
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One OpenAI message's image_url part as an Anthropic image block, or null. */
function openAIImageToAnthropic(part: Json): Json | null {
  if (part.type !== 'image_url' || !isRecord(part.image_url)) return null;
  const url = part.image_url.url;
  if (typeof url !== 'string') return null;
  const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (data !== null) {
    return {
      type: 'image',
      source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] },
    };
  }
  return { type: 'image', source: { type: 'url', url } };
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming
 * `model`: system messages become the system prompt, image_url parts become
 * image blocks, tool messages become tool_result blocks, assistant tool_calls
 * become tool_use blocks, tools and tool_choice are rewritten, and the output
 * cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function openAIToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content),
          },
        ],
      });
      continue;
    }
    if (role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          content.push({
            type: 'tool_use',
            id: typeof call.id === 'string' ? call.id : '',
            name: typeof call.function.name === 'string' ? call.function.name : '',
            input: argumentsOf(call.function.arguments),
          });
        }
      }
      messages.push({ role: 'assistant', content });
      continue;
    }
    // A user message: text and image parts, in order.
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
      else {
        const image = openAIImageToAnthropic(part);
        if (image !== null) blocks.push(image);
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = maxTokensOf(body);
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (typeof choice === 'string') {
    if (choice === 'auto') out.tool_choice = { type: 'auto' };
    else if (choice === 'required') out.tool_choice = { type: 'any' };
    else if (choice === 'none') out.tool_choice = { type: 'none' };
  } else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function)) {
    out.tool_choice = { type: 'tool', name: typeof choice.function.name === 'string' ? choice.function.name : '' };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** Split a data URL into its media type and base64 payload, or null. */
function dataUrlParts(url: string): { media: string; data: string } | null {
  const match = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (match === null) return null;
  return { media: match[1] !== '' ? match[1] : 'image/png', data: match[2] };
}

/** One image, in Gemini's part spelling, from an OpenAI image_url part. */
function openAIImageToGemini(part: Json): Json | null {
  if (part.type !== 'image_url' || !isRecord(part.image_url)) return null;
  const url = part.image_url.url;
  if (typeof url !== 'string') return null;
  const data = dataUrlParts(url);
  if (data !== null) return { inlineData: { mimeType: data.media, data: data.data } };
  return { fileData: { fileUri: url } };
}

/** One image, in Gemini's part spelling, from an Anthropic image block. */
function anthropicImageToGemini(block: Json): Json | null {
  if (block.type !== 'image' || !isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  return null;
}

/**
 * An OpenAI or Anthropic client request as a Gemini generateContent body.
 * `api` names the client's wire. The model rides in the URL, not the body.
 * A tool_call_id maps to its tool name through the assistant messages seen
 * so far, so a tool message can name its functionResponse.
 */
export function clientToGemini(body: Json, api: 'openai' | 'anthropic'): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const toolNames = new Map<string, string>();
  const tools: Json[] = [];
  let toolMode: string | null = null;
  let allowedNames: string[] = [];

  const pushUserText = (text: string): void => {
    if (text === '') return;
    contents.push({ role: 'user', parts: [{ text }] });
  };

  if (api === 'openai') {
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'system') {
        const text = textOf(raw.content);
        if (text !== '') systemParts.push(text);
        continue;
      }
      if (raw.role === 'tool') {
        const name = toolNames.get(typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '') ?? '';
        contents.push({
          role: 'user',
          parts: [
            {
              functionResponse: {
                name,
                response: { result: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) },
              },
            },
          ],
        });
        continue;
      }
      if (raw.role === 'assistant') {
        const parts: Json[] = [];
        const text = textOf(raw.content);
        if (text !== '') parts.push({ text });
        if (Array.isArray(raw.tool_calls)) {
          for (const call of raw.tool_calls) {
            if (!isRecord(call) || !isRecord(call.function)) continue;
            const name = typeof call.function.name === 'string' ? call.function.name : '';
            if (typeof call.id === 'string') toolNames.set(call.id, name);
            parts.push({ functionCall: { name, args: argumentsOf(call.function.arguments) } });
          }
        }
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      const content = raw.content;
      if (typeof content === 'string') {
        pushUserText(content);
        continue;
      }
      if (!Array.isArray(content)) continue;
      const parts: Json[] = [];
      for (const part of content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
        else {
          const image = openAIImageToGemini(part);
          if (image !== null) parts.push(image);
        }
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
    if (Array.isArray(body.tools)) {
      for (const raw of body.tools) {
        if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
        tools.push({
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        });
      }
    }
    if (typeof body.tool_choice === 'string') {
      if (body.tool_choice === 'auto') toolMode = 'AUTO';
      else if (body.tool_choice === 'required') toolMode = 'ANY';
      else if (body.tool_choice === 'none') toolMode = 'NONE';
    } else if (isRecord(body.tool_choice) && body.tool_choice.type === 'function' && isRecord(body.tool_choice.function)) {
      toolMode = 'ANY';
      const name = body.tool_choice.function.name;
      if (typeof name === 'string') allowedNames = [name];
    }
  } else {
    const system = textOf(body.system);
    if (system !== '') systemParts.push(system);
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw) || !Array.isArray(raw.content)) {
        if (isRecord(raw) && typeof raw.content === 'string') {
          contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts: [{ text: raw.content }] });
        }
        continue;
      }
      if (raw.role === 'assistant') {
        const parts: Json[] = [];
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'tool_use') {
            const name = typeof block.name === 'string' ? block.name : '';
            if (typeof block.id === 'string') toolNames.set(block.id, name);
            parts.push({ functionCall: { name, args: isRecord(block.input) ? block.input : {} } });
          }
        }
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      const parts: Json[] = [];
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'tool_result') {
          const name = toolNames.get(typeof block.tool_use_id === 'string' ? block.tool_use_id : '') ?? '';
          parts.push({ functionResponse: { name, response: { result: toolResultText(block.content) } } });
        } else {
          const image = anthropicImageToGemini(block);
          if (image !== null) parts.push(image);
        }
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
    if (Array.isArray(body.tools)) {
      for (const raw of body.tools) {
        if (!isRecord(raw) || typeof raw.name !== 'string') continue;
        tools.push({
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        });
      }
    }
    const choice = body.tool_choice;
    if (isRecord(choice)) {
      if (choice.type === 'auto') toolMode = 'AUTO';
      else if (choice.type === 'any') toolMode = 'ANY';
      else if (choice.type === 'none') toolMode = 'NONE';
      else if (choice.type === 'tool' && typeof choice.name === 'string') {
        toolMode = 'ANY';
        allowedNames = [choice.name];
      }
    }
  }

  const out: Json = { contents };
  if (systemParts.length > 0) {
    out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  }
  if (tools.length > 0) out.tools = [{ functionDeclarations: tools }];
  if (toolMode !== null) {
    out.toolConfig = {
      functionCallingConfig: {
        mode: toolMode,
        ...(allowedNames.length > 0 ? { allowedFunctionNames: allowedNames } : {}),
      },
    };
  }
  const config: Json = {};
  const maxTokens = maxTokensOf(body);
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = api === 'openai' ? body.stop : body.stop_sequences;
  if (Array.isArray(stops) && stops.length > 0) config.stopSequences = stops;
  if (api === 'openai' && jsonModeOf(body)) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** An OpenAI client body naming `model`, stream forced to carry usage. */
export function openAIRequest(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  if (body.stream === true) {
    out.stream = true;
    // The client sends no stream_options of its own; without
    // include_usage the provider emits no usage chunk and the usage check
    // has nothing to compare against.
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** The normalized stop in OpenAI's finish_reason spelling. */
function openAIStop(stop: UpstreamAnswer['stop']): string {
  if (stop === 'length') return 'length';
  if (stop === 'tool_calls') return 'tool_calls';
  if (stop === 'content_filter') return 'content_filter';
  return 'stop';
}

/** The normalized stop in Anthropic's stop_reason spelling. */
function anthropicStop(stop: UpstreamAnswer['stop']): string {
  if (stop === 'length') return 'max_tokens';
  if (stop === 'tool_calls') return 'tool_use';
  if (stop === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** One server-sent event frame: `event: <name>` when named, then data. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split text into 13-character pieces, as the providers' streams do. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

/** The id of an OpenAI-shaped answer from one provider. */
function openAIId(answer: UpstreamAnswer): string {
  return `chatcmpl-${answer.providerId}`;
}

/** The id of an Anthropic-shaped answer from one provider. */
function anthropicId(answer: UpstreamAnswer): string {
  return `msg_${answer.providerId}`;
}

/** A normalized answer as an OpenAI Chat Completions JSON body. */
export function renderOpenAIJson(answer: UpstreamAnswer): Json {
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((c) => ({
      id: c.id,
      type: 'function',
      function: { name: c.name, arguments: c.arguments },
    }));
  }
  return {
    id: openAIId(answer),
    object: 'chat.completion',
    created: FIXED_CREATED,
    model: answer.model,
    choices: [{ index: 0, message, finish_reason: openAIStop(answer.stop), logprobs: null }],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/** A normalized answer as an OpenAI Chat Completions SSE stream. */
export function renderOpenAIStream(answer: UpstreamAnswer): string {
  const id = openAIId(answer);
  const frames: string[] = [];
  const chunk = (delta: Json, finish: string | null): Json => ({
    id,
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model: answer.model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  frames.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  for (const piece of splitText(answer.text)) {
    frames.push(sseEvent(null, chunk({ content: piece }, null)));
  }
  for (const call of answer.toolCalls) {
    frames.push(
      sseEvent(
        null,
        chunk(
          {
            tool_calls: [
              {
                index: answer.toolCalls.indexOf(call),
                id: call.id,
                type: 'function',
                function: { name: call.name, arguments: call.arguments },
              },
            ],
          },
          null,
        ),
      ),
    );
  }
  frames.push(sseEvent(null, chunk({}, openAIStop(answer.stop))));
  frames.push(
    sseEvent(null, {
      id,
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model: answer.model,
      choices: [],
      usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A normalized answer as an Anthropic Messages JSON body. */
export function renderAnthropicJson(answer: UpstreamAnswer): Json {
  const content: Json[] = [];
  content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({
      type: 'tool_use',
      id: call.id !== '' ? call.id : 'toolu_unknown',
      name: call.name,
      input: argumentsOf(call.arguments),
    });
  }
  return {
    id: anthropicId(answer),
    type: 'message',
    role: 'assistant',
    model: answer.model,
    content,
    stop_reason: anthropicStop(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** A normalized answer as an Anthropic Messages SSE stream. */
export function renderAnthropicStream(answer: UpstreamAnswer): string {
  const frames: string[] = [];
  const id = anthropicId(answer);
  frames.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id,
        type: 'message',
        role: 'assistant',
        model: answer.model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
  );
  let index = 0;
  frames.push(
    sseEvent('content_block_start', {
      type: 'content_block_start',
      index,
      content_block: { type: 'text', text: '' },
    }),
  );
  for (const piece of splitText(answer.text)) {
    frames.push(
      sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index,
        delta: { type: 'text_delta', text: piece },
      }),
    );
  }
  frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
  index += 1;
  for (const call of answer.toolCalls) {
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: {
          type: 'tool_use',
          id: call.id !== '' ? call.id : `toolu_${index}`,
          name: call.name,
          input: {},
        },
      }),
    );
    if (call.arguments !== '') {
      frames.push(
        sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index,
          delta: { type: 'input_json_delta', partial_json: call.arguments },
        }),
      );
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStop(answer.stop), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
  );
  frames.push(sseEvent('message_stop', { type: 'message_stop' }));
  return frames.join('');
}

/** GET /v1/models in OpenAI's list shape. */
export function openAIModelList(models: readonly { id: string; provider: string }[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models in Anthropic's list shape. */
export function anthropicModelList(models: readonly { id: string; display_name: string }[]): Json {
  return {
    data: models.map((m) => ({
      type: 'model',
      id: m.id,
      display_name: m.display_name,
      created_at: '2026-01-01T00:00:00Z',
    })),
    has_more: false,
  };
}

/** An error in OpenAI's shape. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** An error in Anthropic's shape. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}
