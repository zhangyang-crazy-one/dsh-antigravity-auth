/**
 * Regression guard: a signed tool call followed by a signature-bearing empty
 * text frame must not emit an unclosed block. DSH's BlockAssembler drops the
 * whole turn's replay envelope when entries no longer align with emitted
 * blocks, which loses the tool-call signature the next request needs.
 */
import { describe, expect, it } from 'vitest'
import { BlockAssembler } from '@deepseek-ai/dsh-llm'
import { AntigravityAdapter, ANTIGRAVITY_PROVIDER } from '../src/llm-adapter.ts'
import type { HostCredential } from '../src/credential-coordinator.ts'
import type { GenerateOptions } from '@deepseek-ai/dsh-llm'

const credential = (token: string): HostCredential => ({
  accessToken: token,
  refreshToken: 'refresh',
  expiresAt: Date.parse('2030-01-01T00:00:00.000Z'),
  projectId: 'project-id',
})

const SIGNATURE = 'gemini-issued-thought-signature'

function options(): GenerateOptions {
  return {
    provider: ANTIGRAVITY_PROVIDER,
    model: 'antigravity-gemini-3.8-flash',
    messages: [{
      id: 'm1' as never,
      role: 'user',
      source: { kind: 'user' },
      content: [{ type: 'text', text: 'go' }],
    }],
  }
}

interface ReplayView {
  readonly blocks?: Array<{ readonly kind: string; readonly signature?: string }>
}

async function assemble(body: string): Promise<BlockAssembler> {
  const adapter = new AntigravityAdapter({
    auth: { credential: async () => credential('opaque-test-value') },
    transport: { request: async () => new Response(body) },
  })
  const assembler = new BlockAssembler()
  for await (const chunk of adapter.stream(options())) assembler.push(chunk)
  return assembler
}

describe('Antigravity stream and replay alignment', () => {
  it('keeps replay metadata when a signed empty text frame trails a signed tool call', async () => {
    const assembler = await assemble(
      `data: {"response":{"parts":[{"text":"planning","thought":true,"thoughtSignature":"${SIGNATURE}"}]}}\n\n`
      + 'data: {"response":{"parts":[{"functionCall":{"id":"call-sig","name":"bash","args":{"command":"pwd"}}}]}}\n\n'
      + `data: {"response":{"parts":[{"text":"","thoughtSignature":"${SIGNATURE}"}],"finishReason":"STOP"}}\n\n`,
    )

    expect(assembler.blocks().map(block => block.type)).toEqual(['reasoning', 'tool-call'])

    const replay = assembler.replayState as ReplayView | undefined
    expect(replay).toBeDefined()
    expect(replay?.blocks?.map(block => block.kind)).toEqual(['reasoning', 'tool-call'])
    expect(replay?.blocks?.some(block => block.kind === 'tool-call' && block.signature !== undefined)).toBe(true)
  })

  it('never announces a text block for an unsigned content-free trailing frame', async () => {
    const assembler = await assemble(
      'data: {"response":{"parts":[{"text":"planning","thought":true}]}}\n\n'
      + 'data: {"response":{"parts":[{"functionCall":{"id":"call-plain","name":"bash","args":{"command":"pwd"}}}]}}\n\n'
      + 'data: {"response":{"parts":[{"text":""}],"finishReason":"STOP"}}\n\n',
    )

    expect(assembler.blocks().map(block => block.type)).toEqual(['reasoning', 'tool-call'])
    expect(assembler.replayState).toBeDefined()
  })

  it('still assembles an ordinary text answer unchanged', async () => {
    const assembler = await assemble(
      'data: {"response":{"parts":[{"text":"thinking","thought":true}]}}\n\n'
      + 'data: {"response":{"parts":[{"text":"Hello "}]}}\n\n'
      + 'data: {"response":{"parts":[{"text":"world"}],"finishReason":"STOP"}}\n\n',
    )

    expect(assembler.blocks().map(block => block.type)).toEqual(['reasoning', 'text'])
    const text = assembler.blocks().find(block => block.type === 'text') as { text: string }
    expect(text.text).toBe('Hello world')
    expect(assembler.replayState).toBeDefined()
  })
})
