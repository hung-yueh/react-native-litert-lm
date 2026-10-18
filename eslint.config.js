// @ts-check
const tseslint = require("typescript-eslint");
const reactHooks = require("eslint-plugin-react-hooks");

module.exports = tseslint.config(
  ...tseslint.configs.recommended,
  {
    files: ["src/**/*.{ts,tsx}", "example/App.tsx", "example/src/**/*.{ts,tsx}"],
    plugins: { "react-hooks": reactHooks },
    rules: {
      // Classic hook rules only — the v7 "recommended" React Compiler rules
      // flag intentional patterns here (e.g. useModel exposing ref state).
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
    },
  },
  {
    // src/core is the react-native-free entry point (react-native-litert-lm/core):
    // type-only imports are fine, runtime imports of RN or Nitro are not.
    files: ["src/core/**/*.ts"],
    rules: {
      "@typescript-eslint/no-restricted-imports": [
        "error",
        {
          paths: [
            { name: "react-native", allowTypeImports: true },
            { name: "react-native-nitro-modules", allowTypeImports: true },
            { name: "react", allowTypeImports: true },
          ],
          patterns: [
            {
              group: ["../*", "!../specs/*"],
              allowTypeImports: true,
              message: "src/core may only import other core modules (or types from ../specs).",
            },
          ],
        },
      ],
    },
  },
  {
    // React Native asset imports use require().
    files: ["example/App.tsx", "example/src/**/*.{ts,tsx}"],
    rules: {
      "@typescript-eslint/no-require-imports": "off",
    },
  },
  {
    // Test files use loose typing and require() mocks liberally.
    files: ["src/__tests__/**", "src/__mocks__/**"],
    rules: {
      "@typescript-eslint/no-require-imports": "off",
      "@typescript-eslint/no-unused-vars": "off",
    },
  },
  {
    ignores: [
      "lib/**",
      // Every build/ dir in the tree is generated CMake/Gradle output; a
      // top-level-only pattern let android/build/reports/**/*.js fail lint.
      "**/build/**",
      "coverage/**",
      "node_modules/**",
      "nitrogen/**",
      "example/node_modules/**",
      "example/android/**",
      "example/ios/**",
      "scripts/**",
      "app.plugin.js",
      "jest.config.js",
      "eslint.config.js",
      "react-native.config.js",
      "example/metro.config.js",
      "example/app.config.js",
      "example/withLibraryTestSpecs.js",
      "example/check-metro.js",
    ],
  },
);
