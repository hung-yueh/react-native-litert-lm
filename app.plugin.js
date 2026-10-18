/**
 * Expo config plugin for react-native-litert-lm.
 *
 * Ensures correct build settings for the LiteRT-LM native module:
 * - Android: minSdkVersion 26, Kotlin 2.3.0 or newer (the litertlm-android AAR needs it)
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

/**
 * The oldest Kotlin Gradle plugin that reads the litertlm-android AAR's
 * metadata (Kotlin reads one minor version ahead; the 0.18 AAR is 2.4).
 */
const MIN_KOTLIN = '2.3.0';

/** Compare dotted version strings numerically: <0, 0 or >0. */
function compareVersions(a, b) {
  const pa = String(a).split('.').map((n) => parseInt(n, 10) || 0);
  const pb = String(b).split('.').map((n) => parseInt(n, 10) || 0);
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const d = (pa[i] || 0) - (pb[i] || 0);
    if (d !== 0) return d;
  }
  return 0;
}

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

    // Kotlin: the litertlm-android AAR carries newer metadata than React
    // Native's default Kotlin can read. Templates from Expo SDK 58 read the
    // Kotlin Gradle plugin version from this property; raise it to the
    // minimum, never lower an app's newer choice.
    const kotlinProp = props.find((p) => p.key === 'android.kotlinVersion');
    if (!kotlinProp) {
      props.push({ type: 'property', key: 'android.kotlinVersion', value: MIN_KOTLIN });
    } else if (compareVersions(kotlinProp.value, MIN_KOTLIN) < 0) {
      kotlinProp.value = MIN_KOTLIN;
    }

    return config;
  });

  // Android, older templates (Expo SDK 55–57): they hard-code an unversioned
  // Kotlin Gradle plugin classpath instead of reading android.kotlinVersion,
  // so pin it in the project-level build.gradle. Surviving
  // `expo prebuild --clean` is why this lives in the plugin.
  config = withProjectBuildGradle(config, (config) => {
    if (config.modResults.language === 'groovy') {
      config.modResults.contents = config.modResults.contents.replace(
        "classpath('org.jetbrains.kotlin:kotlin-gradle-plugin')",
        `classpath('org.jetbrains.kotlin:kotlin-gradle-plugin:${MIN_KOTLIN}')`
      );
    }

    return config;
  });

  return config;
}

module.exports = withLiteRTLM;
