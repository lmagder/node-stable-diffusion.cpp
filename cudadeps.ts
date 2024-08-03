import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";
import loadBinding from "pkg-prebuilds";
import { fileTypeFromBuffer, fileTypeFromStream } from "file-type";
import xz from "xz-decompress";
import { Stream } from "node:stream";
import { buffer } from "node:stream/consumers";
import decompress from "decompress";
import { ReadableStream } from "node:stream/web";

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
  const type = Buffer.isBuffer(input) ? await fileTypeFromBuffer(input) : await fileTypeFromStream(input);
  if (!type || type.ext !== "xz") {
    return [];
  }
  const tar = decompressTar();
  const inStream = Stream.Readable.toWeb(Buffer.isBuffer(input) ? new Stream.PassThrough().end(input) : Stream.Readable.from(input));
  const xzStream = new xz.XzReadableStream(inStream);
  return tar(Stream.Readable.fromWeb(xzStream as ReadableStream));
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

      const archiveFiles = await decompress(data, { plugins: [decompressTarXz(), decompressUnzip()] }) as (decompress.File & { linkname?: string })[];
      archiveFiles.sort((x,y) => x.type.localeCompare(y.type));
      for (const d of archiveFiles) {
        if (d.path.toLowerCase().includes(fileExt) && !d.path.toLowerCase().includes("/stubs/")) {
          const dest = path.join(resolvedPath, path.basename(d.path));
          console.info(`Writing ${dest}`);
          if (d.type === "file") {
            fs.writeFileSync(dest, d.data, { encoding: "binary", mode: d.mode });
          } else if (d.type === "symlink" && d.linkname) {
            fs.symlinkSync(d.linkname, dest);
          } else if (d.type === "link" && d.linkname) {
            fs.linkSync(d.linkname, dest);
          }
        }
      }
    }

    fs.writeFileSync(downloadMarkerPath, "done");
    console.info(`Done`);
  }
}
