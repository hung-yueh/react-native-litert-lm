/**
 * `react-native-litert-lm/core` must load in plain Node — no React Native,
 * no Nitro. Both are rigged to throw if anything in core imports them at
 * runtime (the ESLint guard on src/core catches it statically too).
 */
import * as fs from "fs";
import * as path from "path";

describe("react-native-litert-lm/core", () => {
  it("loads without react-native or react-native-nitro-modules", () => {
    jest.isolateModules(() => {
      jest.doMock(
        "react-native",
        () => {
          throw new Error("core imported react-native at runtime");
        },
        { virtual: true },
      );
      jest.doMock("react-native-nitro-modules", () => {
        throw new Error("core imported react-native-nitro-modules at runtime");
      });
      const core = require("../core");
      expect(typeof core.estimateMemory).toBe("function");
      expect(typeof core.createStreamEventParser).toBe("function");
      expect(typeof core.resolveVariant).toBe("function");
      expect(new core.LiteRTLMError("MODEL_NOT_LOADED", "x")).toBeInstanceOf(Error);
    });
  });

  it("exposes the subpath through core/package.json", () => {
    const root = path.resolve(__dirname, "../..");
    const pkg = JSON.parse(fs.readFileSync(path.join(root, "core/package.json"), "utf8"));
    expect(pkg.main).toBe("../lib/core/index.js");
    expect(pkg.types).toBe("../lib/core/index.d.ts");
    expect(fs.existsSync(path.join(root, "core", pkg["react-native"]))).toBe(true);
    const files: string[] = require("../../package.json").files;
    expect(files).toContain("core/package.json");
  });
});
