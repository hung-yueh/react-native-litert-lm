// Installs react-native-litert-lm's XCTest spec ('Tests') in the generated
// Podfile, so the react-native-litert-lm-Unit-Tests scheme that runs
// ios/Tests survives `expo prebuild --clean`. Expo's Podfile autolinks the
// library but never asks CocoaPods for its test specs.
const { withDangerousMod } = require("expo/config-plugins");
const fs = require("fs");
const path = require("path");

const ANCHOR = "  config = use_native_modules!(config_command)\n";
const LINE =
  "  # Added by withLibraryTestSpecs.js: the library's XCTest target.\n" +
  "  pod 'react-native-litert-lm', :path => '../..', :testspecs => ['Tests']\n";

module.exports = function withLibraryTestSpecs(config) {
  return withDangerousMod(config, [
    "ios",
    (config) => {
      const podfile = path.join(config.modRequest.platformProjectRoot, "Podfile");
      const contents = fs.readFileSync(podfile, "utf8");
      if (contents.includes(":testspecs => ['Tests']")) return config;
      if (!contents.includes(ANCHOR)) {
        throw new Error("withLibraryTestSpecs: the Podfile no longer has the use_native_modules! line to anchor on");
      }
      fs.writeFileSync(podfile, contents.replace(ANCHOR, ANCHOR + LINE));
      return config;
    },
  ]);
};
