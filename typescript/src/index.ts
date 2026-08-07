/**
 * autobatcher – Drop-in OpenAI client for flex and batch inference.
 *
 * Usage:
 *   import { BatchOpenAI } from "autobatcher";   // flex polling by default
 *   import { AsyncOpenAI } from "autobatcher";   // equivalent compatibility name
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
