import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";
import loadBinding from "pkg-prebuilds";
import { isStream } from "is-stream";
import { fileTypeFromStream } from "file-type";
import xz from "xz-decompress";
import { Stream } from "node:stream";
import { buffer } from "node:stream/consumers";
import decompress from "decompress";

const require = createRequire(import.meta.url);
const decompressTar = require("decompress-tar");
const decompressUnzip = require("decompress-unzip");

const bindingOptionsFile = "binding-options.cjs";
const components = ["libcublas", "cuda_cudart"];

const versionListFile = "cuda_version.json";
const downloadMarkerFile = "cudadeps.done";
const repoUrl = "https://developer.download.nvidia.com/compute/cuda/redist";

const nodeToCudaArch = {
  arm64: "aarch64",
  ppc64: "ppc64le",
  x64: "x86_64",
} as Record<string, string>;

const nodeToCudaPlatform = {
  linux: "linux",
  win32: "windows",
} as Record<string, string>;

const decompressTarXz = () => async (input: Buffer | NodeJS.ReadableStream) => {
  let inStream: NodeJS.ReadableStream | undefined;
  if (Buffer.isBuffer(input)) {
    const wrapper = new Stream.PassThrough();
    wrapper.end(input);
    inStream = wrapper;
  } else if (isStream(input)) {
    inStream = input;
  } else {
    throw new TypeError(`Expected a Buffer or Stream, got ${typeof input}`);
  }

  const type = await fileTypeFromStream(inStream);
  if (!type || type.ext !== "xz") {
    return [];
  }

  return decompressTar()(new xz.XzReadableStream(inStream));
};

const arch = nodeToCudaArch[process.env.npm_config_arch || os.arch()];
const platform = nodeToCudaPlatform[process.env.npm_config_platform || os.platform()];
const cudaSubfolder = `${platform}-${arch}`;
const archiveExt = platform === "windows" ? "zip" : "tar.xz";
const fileExt = platform === "windows" ? ".dll" : ".so";

const options = require(path.join(process.cwd(), bindingOptionsFile));
// Find the correct bindings file
const resolvedPath = path.dirname((loadBinding as any).resolve(process.cwd(), options, false, true) as string);

const versionListPath = path.join(resolvedPath, versionListFile);
if (fs.existsSync(versionListPath)) {
  const downloadMarkerPath = path.join(resolvedPath, downloadMarkerFile);
  if (!fs.existsSync(downloadMarkerPath)) {
    console.info(`Downloading components ${components} for ${arch} - ${platform}`);
    const versionList = JSON.parse(fs.readFileSync(versionListPath, { encoding: "utf8" }));
    for (const componentId of components) {
      const componentVersion = versionList[componentId].version;
      const archivePath = `${repoUrl}/${componentId}/${cudaSubfolder}/${componentId}-${cudaSubfolder}-${componentVersion}-archive.${archiveExt}`;
      console.info(`Downloading ${archivePath}...`);
      const file = await fetch(archivePath);
      if (!file.body || !file.ok) throw new Error(`Downloading ${archivePath} failed`);
      const data = await buffer(file.body);
      console.info(`Done.`);
      console.info(`Extracting...`);

      const archiveFiles = await decompress(data, { plugins: [decompressTarXz(), decompressUnzip()] });
      for (const d of archiveFiles) {
        if (d.type === "file" && path.extname(d.path).toLowerCase() === fileExt) {
          const dest = path.join(resolvedPath, path.basename(d.path));
          console.info(`Writing ${dest}`);
          fs.writeFileSync(dest, d.data, { encoding: "binary", mode: d.mode });
        }
      }
    }

    fs.writeFileSync(downloadMarkerPath, "done");
    console.info(`Done`);
  }
}
