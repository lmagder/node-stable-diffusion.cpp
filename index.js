import { createRequire } from "node:module";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
export default require("pkg-prebuilds")(dirname(fileURLToPath(import.meta.url)), require("./binding-options.cjs"));
