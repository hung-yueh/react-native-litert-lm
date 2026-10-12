import { extractFileName, resolveModelFileName } from "../core/modelPath";

describe("modelPath", () => {
  it("strips query strings from URLs", () => {
    expect(extractFileName("https://example.com/model.litertlm?token=abc")).toBe(
      "model.litertlm",
    );
  });

  it("resolveModelFileName uses basename for paths and URLs", () => {
    expect(resolveModelFileName("/data/models/foo.litertlm")).toBe("foo.litertlm");
    expect(resolveModelFileName("bare-name.bin")).toBe("bare-name.bin");
  });
});
