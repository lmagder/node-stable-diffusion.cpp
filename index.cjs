const nativeModule = require("pkg-prebuilds")(__dirname, require("./binding-options.cjs"));
module.exports = nativeModule;
