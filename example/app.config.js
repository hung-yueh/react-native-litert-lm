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
          buildReactNativeFromSource: true,
          useHermesV1: true,
          android: {
            minSdkVersion: 26,
            compileSdkVersion: 36,
            targetSdkVersion: 36
          },
          ios: {
            deploymentTarget: "15.1"
          }
        }
      ],
      "react-native-litert-lm",
      "./withGlogModulemapFix"
    ]
  }
};
