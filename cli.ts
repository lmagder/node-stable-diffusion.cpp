import { parseArgs } from "node:util";
import * as fs from "node:fs";
import { PNG } from "pngjs";
import sd from "node-stable-diffusion.cpp";

const main = async () => {
  const args = parseArgs({
    options: {
      model: { type: "string", short: "m" },
      prompt: { type: "string", short: "p" },
      output: { type: "string", short: "o" },
    },
  });

  if (!args.values.model) {
    console.error("Missing model param");
    process.exit(1);
  }

  const model = args.values.model;
  const prompt = args.values.prompt ?? "A picture of a dog";
  const output = args.values.output ?? "out";

  const ctx = await sd.createContext({ model }, (level, text) => console[level](text));

  const images = await ctx.txt2img({ prompt, batchCount: 2 });
  for (const [idx, img] of images.entries()) {
    const fname = `${output}_${idx}.png`;
    const p = new PNG({ width: img.width, height: img.height, inputHasAlpha: img.channel === 4, colorType: 2 });
    p.data = img.data;
    p.pack().pipe(fs.createWriteStream(fname));
    console.info(`Wrote ${fname}`);
  }

  ctx.dispose();
};

main();
