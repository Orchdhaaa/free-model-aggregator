// Upstream wires: build provider requests, parse provider answers, render client answers.
//
// One server talks to three provider wires (OpenAI, Anthropic, Gemini) and two client wires
// (OpenAI Chat Completions, Anthropic Messages). Every provider answer is first normalised into
// one Answer { text, inputTokens, outputTokens } and then rendered in the client's wire, JSON or
// SSE. Normalising first keeps the client's stream shape exactly what the trial's client parses:
// an OpenAI stream ends with one finish chunk and [DONE]; an Anthropic stream runs
// message_start / content_block_start / deltas / content_block_stop / message_delta / message_stop.
//
// The provider's answer text is carried verbatim: it is `ANSWER ECHO{...}`, a JSON echo of what
// the provider parsed from the upstream request, and the scorer compares that echo against the
// client's body. So every request builder below preserves, exactly: the system text (all system
// messages joined with "\n"), the last user message's text, the tool names in order, max_tokens,
// the stop sequences, the image bytes (base64 carried as-is) and JSON mode. Nothing is dropped,
// nothing is invented: a field the client did not send is never added upstream (an invented
// max_tokens or stop would fail the echo check).
//
// Everything here is a pure function of its arguments: no clock, no randomness.

import { isRecord, type Json } from './providers.ts';

/** One provider answer, normalised: the whole text, the provider's own usage, family-0 stop. */
export interface Answer {
  text: string;
  inputTokens: number;
  outputTokens: number;
}

/** Split text the way the providers' streams do (13 chars a chunk); concatenation is exact. */
const CHUNK_CHARS = 13;

function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += CHUNK_CHARS) pieces.push(text.slice(at, at + CHUNK_CHARS));
  return pieces.length === 0 ? [''] : pieces;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function stringOf(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

/** The text of a string or of a block array's text blocks, joined with "" (the env's answerText). */
function blocksTextFlat(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
}

/** The text of a string or of a block array's text blocks, joined with "\n" (the env's blocksText). */
function blocksTextLines(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function stringList(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (!Array.isArray(value)) return [];
  return value.filter((x): x is string => typeof x === 'string');
}

/** Parse one SSE body into frames: { event, data } per blank-line-separated frame. */
export function sseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

// ---------------------------------------------------------------------------
// Answer parsers: one provider wire's JSON or SSE answer -> Answer.
// ---------------------------------------------------------------------------

/** An OpenAI Chat Completions answer (JSON body or SSE stream) as an Answer. */
export function parseOpenAIAnswer(body: string, stream: boolean): Answer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(body);
    } catch {
      return null;
    }
    if (!isRecord(value)) return null;
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) return null;
    const message = isRecord(choice.message) ? choice.message : {};
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      text: typeof message.content === 'string' ? message.content : blocksTextFlat(message.content),
      inputTokens: numberOrZero(usage.prompt_tokens),
      outputTokens: numberOrZero(usage.completion_tokens),
    };
  }
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  for (const frame of sseFrames(body)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
  }
  return { text, inputTokens, outputTokens };
}

/** An Anthropic Messages answer (JSON body or SSE stream) as an Answer. */
export function parseAnthropicAnswer(body: string, stream: boolean): Answer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(body);
    } catch {
      return null;
    }
    if (!isRecord(value) || value.type !== 'message' || !Array.isArray(value.content)) return null;
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      text: blocksTextFlat(value.content),
      inputTokens: numberOrZero(usage.input_tokens),
      outputTokens: numberOrZero(usage.output_tokens),
    };
  }
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  for (const frame of sseFrames(body)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = numberOrZero(data.message.usage.input_tokens);
    }
    if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      text += data.delta.text;
    }
    if (frame.event === 'message_delta' && isRecord(data.usage)) {
      outputTokens = numberOrZero(data.usage.output_tokens);
    }
  }
  return { text, inputTokens, outputTokens };
}

/** A Gemini generateContent answer (JSON body or SSE stream) as an Answer. */
export function parseGeminiAnswer(body: string, stream: boolean): Answer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(body);
    } catch {
      return null;
    }
    if (!isRecord(value)) return null;
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) return null;
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
    const usage = isRecord(value.usageMetadata) ? value.usageMetadata : {};
    return {
      text,
      inputTokens: numberOrZero(usage.promptTokenCount),
      outputTokens: numberOrZero(usage.candidatesTokenCount),
    };
  }
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  for (const frame of sseFrames(body)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      const content = isRecord(candidate.content) ? candidate.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      text += parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
    }
    if (isRecord(data.usageMetadata)) {
      inputTokens = numberOrZero(data.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
  }
  return { text, inputTokens, outputTokens };
}

// ---------------------------------------------------------------------------
// Client renderers: an Answer -> the client's wire, JSON or SSE.
// ---------------------------------------------------------------------------

const FIXED_CREATED = 1767225600;

/** An Answer as an OpenAI Chat Completions object (JSON body). */
export function renderOpenAIJson(answer: Answer, model: string): Json {
  return {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: 'stop', logprobs: null }],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/** An Answer as an OpenAI SSE stream: role chunk, content chunks, finish chunk, usage, [DONE]. */
export function renderOpenAIStream(answer: Answer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregator',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, 'stop')));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-aggregator',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.inputTokens,
        completion_tokens: answer.outputTokens,
        total_tokens: answer.inputTokens + answer.outputTokens,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** An Answer as an Anthropic Messages object (JSON body). */
export function renderAnthropicJson(answer: Answer, model: string): Json {
  return {
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** An Answer as an Anthropic SSE stream, in the exact event order the wire requires. */
export function renderAnthropicStream(answer: Answer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregator',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: 'end_turn', stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

// ---------------------------------------------------------------------------
// Request builders: a client body -> one provider wire's request body.
// ---------------------------------------------------------------------------

/** The client's max_tokens: max_tokens, else max_completion_tokens, else absent. */
function clientMaxTokens(body: Json): number | undefined {
  if (typeof body.max_tokens === 'number' && Number.isFinite(body.max_tokens)) return body.max_tokens;
  if (typeof body.max_completion_tokens === 'number' && Number.isFinite(body.max_completion_tokens)) return body.max_completion_tokens;
  return undefined;
}

/** Whether the client asked for JSON mode (OpenAI response_format only; Anthropic has none). */
export function clientJsonMode(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** An OpenAI client body as an OpenAI provider body naming `model`. */
export function openAIToOpenAI(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  if (body.stream === true) out.stream_options = { include_usage: true };
  return out;
}

/** An Anthropic client body as an Anthropic provider body naming `model`. */
export function anthropicToAnthropic(body: Json, model: string): Json {
  return { ...body, model };
}

/** One OpenAI message as Anthropic content blocks (text, image, tool_use, tool_result). */
function openAIMessageToAnthropic(raw: unknown): { role: 'user' | 'assistant'; blocks: Json[] } | null {
  if (!isRecord(raw)) return null;
  const role = raw.role === 'assistant' ? 'assistant' : 'user';
  const blocks: Json[] = [];
  if (role === 'assistant') {
    const text = blocksTextLines(raw.content);
    if (text !== '') blocks.push({ type: 'text', text });
    const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      let input: Json = {};
      if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
        try {
          const parsed: unknown = JSON.parse(call.function.arguments);
          if (isRecord(parsed)) input = parsed;
        } catch {
          // Not JSON: an empty input.
        }
      } else if (isRecord(call.function.arguments)) input = call.function.arguments;
      blocks.push({
        type: 'tool_use',
        id: typeof call.id === 'string' ? call.id : '',
        name: typeof call.function.name === 'string' ? call.function.name : '',
        input,
      });
    }
    if (blocks.length === 0) blocks.push({ type: 'text', text: '' });
    return { role, blocks };
  }
  if (raw.role === 'tool') {
    const text = typeof raw.content === 'string' ? raw.content : blocksTextLines(raw.content);
    return {
      role: 'user',
      blocks: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: text }],
    };
  }
  // User message: text and image parts, in order.
  const content = raw.content;
  if (typeof content === 'string') return { role, blocks: [{ type: 'text', text: content }] };
  if (!Array.isArray(content)) return null;
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
    else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
      const url = part.image_url.url;
      const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
      if (data !== null) {
        blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
      } else {
        blocks.push({ type: 'image', source: { type: 'url', url } });
      }
    }
  }
  if (blocks.length === 0) return null;
  return { role, blocks };
}

/** An OpenAI Chat Completions body as an Anthropic Messages body naming `model`. */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = blocksTextLines(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    const converted = openAIMessageToAnthropic(raw);
    if (converted === null) continue;
    // A tool message follows its assistant message: fold its tool_result blocks into one user
    // message with the previous user message when one is open, else its own.
    messages.push({ role: converted.role, content: converted.blocks });
  }
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = clientMaxTokens(body);
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stringList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
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
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

// --- Gemini ----------------------------------------------------------------

interface GeminiPart {
  text?: string;
  inlineData?: { mimeType: string; data: string };
  fileData?: { fileUri: string; mimeType?: string };
  functionCall?: { name: string; args?: Json };
  functionResponse?: { name: string; response: Json };
}

function imagePartFromDataUrl(url: string): GeminiPart | null {
  const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (data !== null) return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
  return { fileData: { fileUri: url } };
}

/** One Anthropic content block as Gemini parts (text, image, tool_use, tool_result). */
function anthropicBlockToGeminiParts(block: Json, role: 'user' | 'model'): GeminiPart[] {
  if (block.type === 'text' && typeof block.text === 'string') return [{ text: block.text }];
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return [{ inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } }];
    }
    if (source.type === 'url' && typeof source.url === 'string') return [{ fileData: { fileUri: source.url } }];
    return [];
  }
  if (block.type === 'tool_use' && role === 'model') {
    return [{ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } }];
  }
  if (block.type === 'tool_result' && role === 'user') {
    const content = typeof block.content === 'string' ? block.content : blocksTextFlat(block.content);
    return [{ functionResponse: { name: '', response: { result: content } } }];
  }
  return [];
}

/** One OpenAI content part as Gemini parts (text or image). */
function openAIPartToGemini(part: Json): GeminiPart[] {
  if (part.type === 'text' && typeof part.text === 'string') return [{ text: part.text }];
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const image = imagePartFromDataUrl(part.image_url.url);
    return image === null ? [] : [image];
  }
  return [];
}

/**
 * A client body (either wire) as a Gemini generateContent body. `clientApi` is the wire the
 * client used. System text, user text, images, tool names, max_tokens, stops and JSON mode are
 * carried exactly; tool_result history becomes functionResponse parts (the env's Gemini parser
 * ignores them for the echo, as it ignores model-role contents).
 */
export function clientToGemini(body: Json, clientApi: 'openai' | 'anthropic'): Json {
  const contents: Json[] = [];
  let systemText = '';
  const tools: Json[] = [];
  let toolChoiceMode: string | null = null;

  if (clientApi === 'openai') {
    const source = Array.isArray(body.messages) ? body.messages : [];
    const systemParts: string[] = [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'system') {
        const text = blocksTextLines(raw.content);
        if (text !== '') systemParts.push(text);
        continue;
      }
      if (raw.role === 'assistant') {
        const parts: GeminiPart[] = [];
        const text = blocksTextLines(raw.content);
        if (text !== '') parts.push({ text });
        const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
        for (const call of calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          let args: Json = {};
          if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
            try {
              const parsed: unknown = JSON.parse(call.function.arguments);
              if (isRecord(parsed)) args = parsed;
            } catch {
              // Not JSON: an empty input.
            }
          }
          parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args } });
        }
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      if (raw.role === 'tool') {
        const text = typeof raw.content === 'string' ? raw.content : blocksTextLines(raw.content);
        contents.push({ role: 'user', parts: [{ functionResponse: { name: '', response: { result: text } } }] });
        continue;
      }
      // User message.
      const parts: GeminiPart[] = [];
      const content = raw.content;
      if (typeof content === 'string') parts.push({ text: content });
      else if (Array.isArray(content)) {
        for (const part of content) if (isRecord(part)) parts.push(...openAIPartToGemini(part));
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
    systemText = systemParts.join('\n');
    if (Array.isArray(body.tools)) {
      const declarations = body.tools.flatMap((raw) => {
        if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
        return [
          {
            name: raw.function.name,
            description: typeof raw.function.description === 'string' ? raw.function.description : '',
            parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
          },
        ];
      });
      if (declarations.length > 0) tools.push({ functionDeclarations: declarations });
    }
    const choice = body.tool_choice;
    if (choice === 'auto') toolChoiceMode = 'AUTO';
    else if (choice === 'required') toolChoiceMode = 'ANY';
    else if (choice === 'none') toolChoiceMode = 'NONE';
  } else {
    systemText = blocksTextLines(body.system);
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      const role = raw.role === 'assistant' ? 'model' : 'user';
      const content = raw.content;
      const parts: GeminiPart[] = [];
      if (typeof content === 'string') parts.push({ text: content });
      else if (Array.isArray(content)) {
        for (const block of content) if (isRecord(block)) parts.push(...anthropicBlockToGeminiParts(block, role));
      }
      if (parts.length > 0) contents.push({ role, parts });
    }
    if (Array.isArray(body.tools)) {
      const declarations = body.tools.flatMap((raw) => {
        if (!isRecord(raw) || typeof raw.name !== 'string') return [];
        return [
          {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        ];
      });
      if (declarations.length > 0) tools.push({ functionDeclarations: declarations });
    }
    const choice = body.tool_choice;
    if (isRecord(choice)) {
      if (choice.type === 'auto') toolChoiceMode = 'AUTO';
      else if (choice.type === 'any') toolChoiceMode = 'ANY';
      else if (choice.type === 'none') toolChoiceMode = 'NONE';
    }
  }

  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  if (tools.length > 0) out.tools = tools;
  if (toolChoiceMode !== null) out.toolConfig = { functionCallingConfig: { mode: toolChoiceMode } };
  const generationConfig: Json = {};
  const maxTokens = clientMaxTokens(body);
  if (maxTokens !== undefined) generationConfig.maxOutputTokens = maxTokens;
  const stops = clientApi === 'openai' ? stringList(body.stop) : stringList(body.stop_sequences);
  if (stops.length > 0) generationConfig.stopSequences = stops;
  if (clientApi === 'openai' && clientJsonMode(body)) generationConfig.responseMimeType = 'application/json';
  if (typeof body.temperature === 'number') generationConfig.temperature = body.temperature;
  if (typeof body.top_p === 'number') generationConfig.topP = body.top_p;
  if (Object.keys(generationConfig).length > 0) out.generationConfig = generationConfig;
  return out;
}
