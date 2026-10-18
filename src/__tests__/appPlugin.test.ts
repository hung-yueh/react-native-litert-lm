// The Expo config plugin is plain CommonJS at the package root.
const withLiteRTLM = require('../../app.plugin.js');

const EVA = 'com.apple.developer.kernel.extended-virtual-addressing';
const IML = 'com.apple.developer.kernel.increased-memory-limit';

/** Run the plugin's iOS entitlements mod (if it added one) over `existing`. */
async function entitlementsAfter(options?: object, existing: Record<string, unknown> = {}) {
  const config = withLiteRTLM({ name: 'app', slug: 'app' }, options);
  const mod = config.mods?.ios?.entitlements;
  if (!mod) return null;
  const result = await mod({
    ...config,
    modResults: { ...existing },
    modRequest: { platform: 'ios', modName: 'entitlements', projectRoot: '/tmp', platformProjectRoot: '/tmp/ios', introspect: false },
  });
  return result.modResults as Record<string, unknown>;
}

describe('Expo config plugin', () => {
  it('adds no iOS entitlements by default (they need a paid developer account)', async () => {
    expect(await entitlementsAfter()).toBeNull();
    expect(await entitlementsAfter({ ios: {} })).toBeNull();
  });

  it('adds Extended Virtual Addressing when asked', async () => {
    expect(await entitlementsAfter({ ios: { extendedVirtualAddressing: true } })).toEqual({ [EVA]: true });
  });

  it('adds both memory entitlements and keeps the app’s own', async () => {
    const result = await entitlementsAfter(
      { ios: { extendedVirtualAddressing: true, increasedMemoryLimit: true } },
      { 'aps-environment': 'development' },
    );
    expect(result).toEqual({ 'aps-environment': 'development', [EVA]: true, [IML]: true });
  });

  describe('Android', () => {
    const modRequest = { platform: 'android', projectRoot: '/tmp', platformProjectRoot: '/tmp/android', introspect: false };

    async function gradleProperties(existing: Array<{ type: string; key: string; value: string }>) {
      const config = withLiteRTLM({ name: 'app', slug: 'app' }, { ios: { extendedVirtualAddressing: true } });
      const result = await config.mods.android.gradleProperties({
        ...config, modResults: existing, modRequest: { ...modRequest, modName: 'gradleProperties' },
      });
      return result.modResults.find((p: { key: string }) => p.key === 'android.minSdkVersion')?.value;
    }

    it('raises minSdkVersion to 26, adding it when missing, and leaves a higher one alone', async () => {
      expect(await gradleProperties([])).toBe('26');
      expect(await gradleProperties([{ type: 'property', key: 'android.minSdkVersion', value: '24' }])).toBe('26');
      expect(await gradleProperties([{ type: 'property', key: 'android.minSdkVersion', value: '28' }])).toBe('28');
    });

    async function kotlinVersion(existing: Array<{ type: string; key: string; value: string }>) {
      const config = withLiteRTLM({ name: 'app', slug: 'app' });
      const result = await config.mods.android.gradleProperties({
        ...config, modResults: existing, modRequest: { ...modRequest, modName: 'gradleProperties' },
      });
      return result.modResults.find((p: { key: string }) => p.key === 'android.kotlinVersion')?.value;
    }

    it('sets android.kotlinVersion to at least 2.3.0 (Expo SDK 58 templates read it)', async () => {
      expect(await kotlinVersion([])).toBe('2.3.0');
      expect(await kotlinVersion([{ type: 'property', key: 'android.kotlinVersion', value: '2.2.0' }])).toBe('2.3.0');
      expect(await kotlinVersion([{ type: 'property', key: 'android.kotlinVersion', value: '2.10.1' }])).toBe('2.10.1');
    });

    it('pins the Kotlin Gradle plugin in older templates', async () => {
      const config = withLiteRTLM({ name: 'app', slug: 'app' });
      const result = await config.mods.android.projectBuildGradle({
        ...config,
        modResults: { language: 'groovy', contents: "classpath('org.jetbrains.kotlin:kotlin-gradle-plugin')" },
        modRequest: { ...modRequest, modName: 'projectBuildGradle' },
      });
      expect(result.modResults.contents).toBe("classpath('org.jetbrains.kotlin:kotlin-gradle-plugin:2.3.0')");
    });
  });
});
