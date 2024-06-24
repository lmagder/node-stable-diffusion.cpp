import { parseArgs } from "node:util";
import { mkdir } from "node:fs/promises";
import { dirname } from "node:path";

import sharp from "sharp";
import sd from "node-stable-diffusion-cpp";

const args = parseArgs({
  options: {
    model: { type: "string", short: "m" },
    prompt: { type: "string", short: "p" },
    output: { type: "string", short: "o" },
    width: { type: "string", short: "w" },
    height: { type: "string", short: "h" },
    batchCount: { type: "string", short: "b" },
  },
});

if (!args.values.model) {
  console.error("Missing model param");
  process.exit(1);
}

const model = args.values.model;
const prompt = args.values.prompt ?? "a picture of a dog";
const output = args.values.output ?? `out/${prompt}`;
const width = Number.parseInt(args.values.width ?? "768");
const height = Number.parseInt(args.values.height ?? "768");
const batchCount = Number.parseInt(args.values.batchCount ?? "1");

const ctx = await sd.createContext({ model }, (level, text) => console[level](text));

const images = await ctx.txt2img({ prompt, batchCount, width, height, sampleMethod: sd.SampleMethod.LCM });
for (const [idx, img] of images.entries()) {
  const fname = `${output}_${idx}.jpg`;
  await mkdir(dirname(fname), { recursive: true });
  await sharp(img.data, { raw: { width: img.width, height: img.height, channels: img.channel } })
    .jpeg()
    .toFile(fname);
  console.info(`Wrote ${fname}`);
}

await ctx.dispose();
