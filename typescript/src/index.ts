/**
 * autobatcher – Drop-in OpenAI client that transparently batches requests.
 *
 * Usage:
 *   import { BatchOpenAI } from "autobatcher";   // 24h batch inference (default)
 *   import { AsyncOpenAI } from "autobatcher";   // 1h async inference
 *
 *   const client = new BatchOpenAI({ apiKey: "..." });
 *   const response = await client.chat.completions.create({
 *     model: "gpt-4o",
 *     messages: [{ role: "user", content: "Hello!" }],
 *   });
 */

export { AsyncOpenAI, BatchOpenAI } from "./client.js";
export type { BatchOpenAIOptions } from "./client.js";
export { serve } from "./serve.js";
export type { ServeOptions } from "./serve.js";
