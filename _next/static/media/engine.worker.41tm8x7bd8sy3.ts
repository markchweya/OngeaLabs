/// <reference lib="webworker" />

/*
  Ongea's speech engine, in a Web Worker so the page never stutters while a
  model loads or speaks.

  Everything runs in the visitor's browser on ONNX Runtime's WebAssembly
  build: no audio and no text ever leaves the device. Models download once
  from Hugging Face (or, for Swahili, from this site) and stay in the
  browser's cache, so the second visit speaks at once.

  Messages in:  { type: 'speak', id, text, voice, speed }
                voice is a base voice id (voices.ts) or { recipe } (blend.ts)
  Messages out: { type: 'progress', id, stage, loaded, total }
                { type: 'chunk', id, audio, sampleRate, offset }
                  one sentence as soon as it is spoken, and where it
                  starts in the whole (in samples), so the page can play
                  the first sentence while the rest are still being made
                { type: 'done', id, audio (Float32Array, transferred), sampleRate }
                { type: 'error', id, message }

  Requests are handled one at a time, in order: two models in memory at
  once is fine, two inferences at once on one thread only makes both slow.
*/

import { env, pipeline, type TextToAudioPipeline } from '@huggingface/transformers'
import { KokoroTTS } from 'kokoro-js'

import { STYLE_ROWS, STYLE_WIDTH, mixStyles, normaliseRecipe, recipeKey, type Recipe } from '@/lib/ongea/blend'
import { asset } from '@/lib/ongea/paths'
import { MAX_TEXT, SENTENCE_GAP_SECONDS, joinWithGaps, splitSentences } from '@/lib/ongea/text'
import { KOKORO_MODEL, MMS_MODELS, baseVoice } from '@/lib/ongea/voices'

export type SpeakRequest = {
  type: 'speak'
  id: number
  text: string
  voice: string | { recipe: Recipe }
  /** 0.5 to 2, 1 is natural. */
  speed: number
}

export type EngineMessage =
  | { type: 'progress'; id: number; stage: 'download' | 'prepare' | 'speak'; loaded: number; total: number }
  | { type: 'chunk'; id: number; audio: Float32Array; sampleRate: number; offset: number }
  | { type: 'done'; id: number; audio: Float32Array; sampleRate: number }
  | { type: 'error'; id: number; message: string }

declare const self: DedicatedWorkerGlobalScope

// ONNX Runtime's WebAssembly comes from this site (copied from node_modules
// at build time, scripts/copy-ort.mjs), not from a CDN. Threads need a
// cross-origin isolated page; without one it runs on one, which is fine.
if (env.backends.onnx.wasm) {
  env.backends.onnx.wasm.wasmPaths = asset('/ongea/ort/')
  env.backends.onnx.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1
}
env.useBrowserCache = true
// Every model is fetched by URL (see loadMms); none come from a local path.
env.allowLocalModels = false
env.allowRemoteModels = true

const VOICE_URL = (id: string) => `https://huggingface.co/${KOKORO_MODEL}/resolve/main/voices/${id}.bin`

/* ---------------------------------------------------------------- Kokoro */

/**
 * KokoroTTS with blended voices. A voice name of the form "blend:<key>"
 * reads its style table from `blends` instead of downloading one; the rest
 * is Kokoro's own pipeline (phonemiser, sentence splitting, model call).
 */
class OngeaKokoro extends KokoroTTS {
  readonly blends = new Map<string, { table: Float32Array; accent: 'a' | 'b' }>()

  _validate_voice(voice: string) {
    const blend = this.blends.get(voice)
    if (blend) return blend.accent
    return super._validate_voice(voice)
  }

  // @ts-expect-error The base signature types voice as Kokoro's names.
  async generate_from_ids(inputIds: Parameters<KokoroTTS['generate_from_ids']>[0], options: { voice: string; speed: number }) {
    const blend = this.blends.get(options.voice)
    // @ts-expect-error As above.
    if (!blend) return super.generate_from_ids(inputIds, options)

    // As Kokoro does: the style row for this many tokens, less the two
    // boundary tokens, capped at the table's last row.
    const { Tensor, RawAudio } = await import('@huggingface/transformers')
    const tokens = inputIds.dims.at(-1) ?? 0
    const row = Math.min(Math.max(tokens - 2, 0), STYLE_ROWS - 1)
    const style = blend.table.slice(row * STYLE_WIDTH, (row + 1) * STYLE_WIDTH)
    const { waveform } = await this.model({
      input_ids: inputIds,
      style: new Tensor('float32', style, [1, STYLE_WIDTH]),
      speed: new Tensor('float32', [options.speed], [1]),
    })
    return new RawAudio(waveform.data as Float32Array, 24000)
  }
}

let kokoro: Promise<OngeaKokoro> | null = null
const voiceTables = new Map<string, Promise<Float32Array>>()

function loadKokoro(report: (loaded: number, total: number) => void) {
  kokoro ??= (async () => {
    const base = await KokoroTTS.from_pretrained(KOKORO_MODEL, {
      dtype: 'q8',
      device: 'wasm',
      progress_callback: progressReporter(report),
    })
    return new OngeaKokoro(base.model, base.tokenizer)
  })().catch(error => {
    kokoro = null
    throw error
  })
  return kokoro
}

function loadVoiceTable(id: string) {
  let table = voiceTables.get(id)
  if (!table) {
    table = (async () => {
      const cache = await caches.open('ongea-voices').catch(() => null)
      const url = VOICE_URL(id)
      const response = (await cache?.match(url)) ?? (await fetch(url))
      if (!response.ok) throw new Error(`voice ${id} did not load (${response.status})`)
      if (cache && !(await cache.match(url))) await cache.put(url, response.clone())
      const data = new Float32Array(await response.arrayBuffer())
      if (data.length !== STYLE_ROWS * STYLE_WIDTH) throw new Error(`voice ${id} has an unexpected size`)
      return data
    })().catch(error => {
      voiceTables.delete(id)
      throw error
    })
    voiceTables.set(id, table)
  }
  return table
}

async function speakKokoro(
  text: string,
  voice: string | { recipe: Recipe },
  speed: number,
  report: Reporter,
  emit: Emitter
) {
  const tts = await loadKokoro((loaded, total) => report('download', loaded, total))

  let voiceName: string
  if (typeof voice === 'string') {
    voiceName = voice
  } else {
    const recipe = normaliseRecipe(voice.recipe)
    voiceName = `blend:${recipeKey(recipe)}`
    if (!tts.blends.has(voiceName)) {
      report('prepare', 0, 1)
      const tables = await Promise.all(recipe.parts.map(part => loadVoiceTable(part.voice)))
      tts.blends.set(voiceName, {
        table: mixStyles(tables, recipe.parts.map(part => part.weight)),
        accent: recipe.accent,
      })
    }
  }

  const sentences = splitSentences(text)
  const chunks: Float32Array[] = []
  for (const [index, sentence] of sentences.entries()) {
    report('speak', index, sentences.length)
    // @ts-expect-error Blended voice names are not in Kokoro's list.
    const audio = await tts.generate(sentence, { voice: voiceName, speed })
    chunks.push(audio.audio as Float32Array)
    emit(chunks, 24000)
  }

  return { audio: joinWithGaps(chunks, 24000), sampleRate: 24000 }
}

/* ------------------------------------------------------------------- MMS */

// pipeline()'s overloads are too wide for TypeScript to resolve here (TS2590).
const loadTextToSpeech = pipeline as unknown as (
  task: 'text-to-speech',
  model: string,
  options: Record<string, unknown>
) => Promise<TextToAudioPipeline>

const mmsPipelines = new Map<string, Promise<TextToAudioPipeline>>()

// Where transformers.js fetches models from. Hugging Face by default; for a
// model this site hosts (MMS_MODELS entries starting with '/') the same
// "remote" path points at this origin instead. Remote rather than local
// mode, because only remote loading treats a missing optional file (VITS
// has no preprocessor_config.json) as "not needed" instead of an error,
// and it caches by URL like every other model.
const HUGGING_FACE = { host: env.remoteHost, template: env.remotePathTemplate }

// Loads are serialised (one request at a time), so switching the global
// host below never races another load.
function loadMms(voice: string, report: (loaded: number, total: number) => void) {
  let loading = mmsPipelines.get(voice)
  if (!loading) {
    const model = MMS_MODELS[voice]
    const ownHosted = model.startsWith('/')
    const id = ownHosted ? model.slice(model.lastIndexOf('/') + 1) : model
    loading = (async () => {
      if (ownHosted) {
        env.remoteHost = `${self.location.origin}/`
        env.remotePathTemplate = `${model.slice(1, model.lastIndexOf('/') + 1)}{model}/`
      }
      try {
        return await loadTextToSpeech('text-to-speech', id, {
          dtype: 'q8',
          device: 'wasm',
          progress_callback: progressReporter(report),
        })
      } finally {
        env.remoteHost = HUGGING_FACE.host
        env.remotePathTemplate = HUGGING_FACE.template
      }
    })().catch(error => {
      mmsPipelines.delete(voice)
      throw error
    })
    mmsPipelines.set(voice, loading)
  }
  return loading
}

async function speakMms(text: string, voice: string, speed: number, report: Reporter, emit: Emitter) {
  const synthesiser = await loadMms(voice, (loaded, total) => report('download', loaded, total))

  const sentences = splitSentences(text)
  const chunks: Float32Array[] = []
  let sampleRate = 16000
  for (const [index, sentence] of sentences.entries()) {
    report('speak', index, sentences.length)
    const output = await synthesiser(sentence, {})
    sampleRate = output.sampling_rate
    chunks.push(output.audio as Float32Array)
    emit(chunks, sampleRate)
  }

  // MMS has no speed input in its browser build; the studio's pace control
  // reshapes its audio afterwards instead (voice-shaper).
  void speed
  return { audio: joinWithGaps(chunks, sampleRate), sampleRate }
}

/* ---------------------------------------------------------------- shared */

type Reporter = (stage: 'download' | 'prepare' | 'speak', loaded: number, total: number) => void
/** Called with all sentences so far, after each new one; sends the newest. */
type Emitter = (chunks: Float32Array[], sampleRate: number) => void

/** Sums transformers.js's per-file download progress into one figure. */
function progressReporter(report: (loaded: number, total: number) => void) {
  const files = new Map<string, { loaded: number; total: number }>()
  return (event: { status: string; file?: string; loaded?: number; total?: number }) => {
    if (event.status !== 'progress' || !event.file) return
    files.set(event.file, { loaded: event.loaded ?? 0, total: event.total ?? 0 })
    let loaded = 0
    let total = 0
    for (const file of files.values()) {
      loaded += file.loaded
      total += file.total
    }
    report(loaded, total)
  }
}

/* ----------------------------------------------------------------- queue */

let queue: Promise<void> = Promise.resolve()

self.addEventListener('message', (event: MessageEvent<SpeakRequest>) => {
  const request = event.data
  if (request?.type !== 'speak') return

  queue = queue.then(async () => {
    const post = (message: EngineMessage, transfer: Transferable[] = []) => self.postMessage(message, transfer)
    const report: Reporter = (stage, loaded, total) => post({ type: 'progress', id: request.id, stage, loaded, total })
    const emit: Emitter = (chunks, sampleRate) => {
      // Where this sentence starts in the joined audio: everything before
      // it plus one gap per sentence before it (joinWithGaps).
      const gap = Math.round(SENTENCE_GAP_SECONDS * sampleRate)
      const offset = chunks.slice(0, -1).reduce((sum, chunk) => sum + chunk.length + gap, 0)
      // A copy goes out; the original stays for the final join.
      const audio = chunks[chunks.length - 1].slice()
      post({ type: 'chunk', id: request.id, audio, sampleRate, offset }, [audio.buffer])
    }

    try {
      const text = String(request.text ?? '').slice(0, MAX_TEXT).trim()
      if (!text) throw new Error('Nothing to say yet.')
      const speed = Math.min(2, Math.max(0.5, Number(request.speed) || 1))

      const isBlend = typeof request.voice === 'object'
      const base = typeof request.voice === 'string' ? baseVoice(request.voice) : undefined
      if (!isBlend && !base) throw new Error('That voice is not available.')

      const result =
        isBlend || base?.engine === 'kokoro'
          ? await speakKokoro(text, request.voice, speed, report, emit)
          : await speakMms(text, base!.id, speed, report, emit)

      post({ type: 'done', id: request.id, audio: result.audio, sampleRate: result.sampleRate }, [result.audio.buffer])
    } catch (error) {
      post({ type: 'error', id: request.id, message: error instanceof Error ? error.message : 'The voice engine failed.' })
    }
  })
})
