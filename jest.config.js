module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  testMatch: ['**/__tests__/**/*.test.ts?(x)'],
  // Ignore the compiled output so its duplicate __mocks__ copy can't shadow
  // the canonical src/__mocks__ in jest's haste map (which caused
  // non-deterministic mock resolution once the haste cache was cleared).
  modulePathIgnorePatterns: ['<rootDir>/lib/'],
  moduleNameMapper: {
    '^react-native-nitro-modules$': '<rootDir>/src/__mocks__/react-native-nitro-modules.ts',
  },
  collectCoverage: true,
  coverageDirectory: 'coverage',
  coverageThreshold: {
    global: {
      branches: 70,
      functions: 85,
      lines: 85,
      statements: 85,
    },
  },
};
