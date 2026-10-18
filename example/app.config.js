module.exports = {
  expo: {
    name: "LLMTest",
    slug: "LLMTest",
    version: "1.0.0",
    orientation: "portrait",
    icon: "./assets/icon.png",
    userInterfaceStyle: "light",
    splash: {
      image: "./assets/splash-icon.png",
      resizeMode: "contain",
      backgroundColor: "#ffffff"
    },
    ios: {
      supportsTablet: true,
      bundleIdentifier: "com.hughchen.LLMTest",
      // Expo SDK 58's floor (expo-modules-core); the library itself supports 15.1.
      deploymentTarget: "16.4",
      entitlements: {
        "com.apple.developer.kernel.increased-memory-limit": true,
        "com.apple.developer.kernel.extended-virtual-addressing": true
      }
    },
    android: {
      adaptiveIcon: {
        foregroundImage: "./assets/adaptive-icon.png",
        backgroundColor: "#ffffff"
      },
      predictiveBackGestureEnabled: false,
      package: "com.hughchen.LLMTest"
    },
    web: {
      favicon: "./assets/favicon.png"
    },
    plugins: [
      [
        "expo-build-properties",
        {
          // Hermes V1 is React Native's default engine (since 0.84), so the
          // prebuilt React Native core works; no source build needed.
          android: {
            minSdkVersion: 26
          }
        }
      ],
      "react-native-litert-lm",
      "./withLibraryTestSpecs"
    ]
  }
};
