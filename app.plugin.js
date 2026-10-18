/**
 * Expo config plugin for react-native-litert-lm.
 *
 * Ensures correct build settings for the LiteRT-LM native module:
 * - Android: minSdkVersion 26, Kotlin 2.3.0 (required by litertlm-android AAR)
 * - iOS (opt-in): the memory entitlements multi-GB models need.
 *
 *   plugins: [["react-native-litert-lm", {
 *     ios: { extendedVirtualAddressing: true, increasedMemoryLimit: true },
 *   }]]
 *
 * Both entitlements need a paid Apple Developer account, so neither is on by
 * default: a free provisioning profile can't sign an app that declares them.
 */
const {
  withEntitlementsPlist,
  withGradleProperties,
  withProjectBuildGradle,
} = require('@expo/config-plugins');

/** Plugin option → entitlement key. */
const IOS_ENTITLEMENTS = {
  // Lifts the ~2 GB virtual-memory cap; without it Jetsam kills the app while
  // loading Gemma 4 or Gemma 3n.
  extendedVirtualAddressing: 'com.apple.developer.kernel.extended-virtual-addressing',
  // Raises the app's memory limit on devices that allow it.
  increasedMemoryLimit: 'com.apple.developer.kernel.increased-memory-limit',
};

function withLiteRTLM(config, options = {}) {
  const ios = (options && options.ios) || {};
  const entitlements = Object.keys(IOS_ENTITLEMENTS).filter((option) => ios[option] === true);
  if (entitlements.length > 0) {
    config = withEntitlementsPlist(config, (config) => {
      for (const option of entitlements) {
        config.modResults[IOS_ENTITLEMENTS[option]] = true;
      }
      return config;
    });
  }

  // Android: Ensure minSdkVersion is at least 26
  config = withGradleProperties(config, (config) => {
    const props = config.modResults;

    // Set minSdkVersion if not already high enough
    const minSdkProp = props.find((p) => p.key === 'android.minSdkVersion');
    if (!minSdkProp) {
      props.push({
        type: 'property',
        key: 'android.minSdkVersion',
        value: '26',
      });
    } else if (parseInt(minSdkProp.value, 10) < 26) {
      minSdkProp.value = '26';
    }

    return config;
  });

  // Android: Pin Kotlin Gradle plugin to 2.3.0
  // The litertlm-android AAR uses Kotlin 2.3.0 metadata (version defined in
  // package.json → litertLm.androidMavenVersion).
  // React Native's default Kotlin version (2.1.0) cannot read this metadata,
  // so we must force the Kotlin Gradle plugin to 2.3.0 in the project-level
  // build.gradle. This ensures the fix survives `expo prebuild --clean`.
  config = withProjectBuildGradle(config, (config) => {
    if (config.modResults.language === 'groovy') {
      const contents = config.modResults.contents;

      // Only add if not already pinned
      if (!contents.includes("kotlin-gradle-plugin:2.3.0")) {
        // Replace the unversioned kotlin-gradle-plugin classpath with a pinned one
        config.modResults.contents = contents.replace(
          "classpath('org.jetbrains.kotlin:kotlin-gradle-plugin')",
          "classpath('org.jetbrains.kotlin:kotlin-gradle-plugin:2.3.0')"
        );
      }
    }

    return config;
  });

  return config;
}

module.exports = withLiteRTLM;
