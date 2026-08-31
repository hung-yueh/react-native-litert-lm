// Fixes the iOS build under Swift C++ interop (nitro 0.37+ / Xcode 26).
//
// CocoaPods generates a glog modulemap with per-header submodules
// (`module * { export * }`), but glog/logging.h textually #includes
// log_severity.h *inside* `namespace google`. That is legal for textual
// includes, but react-native-nitro-modules 0.37+ exposes view-template
// headers (cpp/views/ReactProp.hpp etc.) in its public Swift modulemap,
// which pull React renderer headers -> react_native_assert.h ->
// <glog/logging.h>. Swift's Clang importer must then build glog as a
// module, the namespaced include becomes an illegal submodule import
// ("import of module 'glog.glog.log_severity' appears within namespace
// 'google'"), and the NitroModules SwiftEmitModule step fails.
//
// Excluding log_severity.h / vlog_is_on.h from the module makes every
// include of them textual (they are designed to be textually included
// inside `namespace google`), while module glog still exports logging.h.
//
// Only needed because this app sets `buildReactNativeFromSource: true`
// (for useHermesV1): on RN's default prebuilt path glog is not a pod at
// all (ReactNativeDependencies replaces it) and this plugin is a no-op —
// verified 2026-08-30. Remove it if the app drops source builds, or once
// nitro stops exposing its views headers in the Swift modulemap upstream.
const { withDangerousMod } = require("expo/config-plugins");
const fs = require("fs");
const path = require("path");

const RUBY_PATCH = `    # Patched in by withGlogModulemapFix.js — see that file for the why.
    glog_modulemap = File.join(installer.sandbox.root, 'Target Support Files', 'glog', 'glog.modulemap')
    if File.exist?(glog_modulemap)
      File.write(glog_modulemap, <<~MODULEMAP)
        module glog {
          umbrella header "glog-umbrella.h"
          exclude header "glog/log_severity.h"
          exclude header "glog/vlog_is_on.h"
          export *
        }
      MODULEMAP
    end
`;

module.exports = function withGlogModulemapFix(config) {
  return withDangerousMod(config, [
    "ios",
    (cfg) => {
      const podfilePath = path.join(cfg.modRequest.platformProjectRoot, "Podfile");
      let contents = fs.readFileSync(podfilePath, "utf8");
      if (!contents.includes("withGlogModulemapFix")) {
        contents = contents.replace(
          /post_install do \|installer\|\n/,
          (match) => match + RUBY_PATCH
        );
        fs.writeFileSync(podfilePath, contents);
      }
      return cfg;
    },
  ]);
};
