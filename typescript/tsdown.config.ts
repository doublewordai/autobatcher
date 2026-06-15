import { defineConfig } from "tsdown";

export default defineConfig({
  entry: ["src/index.ts", "src/cli.ts"],
  format: ["esm", "cjs"],
  dts: true,
  sourcemap: true,
  clean: true,
  target: false,
  // Keep tsup's output naming (ESM as .js/.d.ts) so package.json "exports"
  // and the published filenames stay unchanged. CJS keeps tsdown's default
  // .cjs/.d.cts.
  outExtensions: ({ format }) =>
    format === "es" ? { js: ".js", dts: ".d.ts" } : undefined,
});
