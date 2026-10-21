import { ModelRegistry } from '../modelRegistry';
import { mockModelStore } from '../__mocks__/react-native-nitro-modules';

describe('ModelRegistry Unit Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('isCached should query native store correctly', () => {
    mockModelStore.isCached.mockReturnValueOnce(true);
    const result = ModelRegistry.isCached('https://example.com/test-model.litertlm');
    expect(mockModelStore.isCached).toHaveBeenCalledWith('test-model.litertlm');
    expect(result).toBe(true);
  });

  it('getFilePath should return cached path', () => {
    mockModelStore.getFilePath.mockReturnValueOnce('/caches/test.bin');
    const path = ModelRegistry.getFilePath('test.bin');
    expect(mockModelStore.getFilePath).toHaveBeenCalledWith('test.bin');
    expect(path).toBe('/caches/test.bin');
  });

  it('listCachedFiles should delegate to native', () => {
    const mockFiles = [
      {
        fileName: 'model.bin',
        absolutePath: '/caches/model.bin',
        sizeBytes: 1000,
        lastModifiedMs: 12345,
      },
    ];
    mockModelStore.listCachedFiles.mockReturnValueOnce(mockFiles as any);
    const files = ModelRegistry.listCachedFiles();
    expect(mockModelStore.listCachedFiles).toHaveBeenCalled();
    expect(files).toEqual(mockFiles);
  });

  it('deleteFile should delegate delete to native', () => {
    ModelRegistry.deleteFile('https://example.com/model.bin?q=1');
    expect(mockModelStore.deleteFile).toHaveBeenCalledWith('model.bin');
  });

  it('resolveModel should throw error on HTTP urls', async () => {
    await expect(ModelRegistry.resolveModel('http://example.com/model.bin'))
      .rejects.toThrow('Insecure HTTP URLs are not allowed for model downloads');
  });

  it('resolveModel should download HTTPS urls', async () => {
    mockModelStore.downloadFile.mockResolvedValueOnce('/downloaded/model.bin');
    const path = await ModelRegistry.resolveModel('https://example.com/model.bin', {
      headers: { Authorization: 'Bearer test' },
    });
    expect(mockModelStore.downloadFile).toHaveBeenCalledWith(
      'https://example.com/model.bin',
      'model.bin',
      JSON.stringify({ Authorization: 'Bearer test' }),
      '',
      expect.any(Function)
    );
    expect(path).toBe('/downloaded/model.bin');
  });

  it('resolveModel should return local paths directly', async () => {
    const path = await ModelRegistry.resolveModel('/local/path/model.bin');
    expect(mockModelStore.downloadFile).not.toHaveBeenCalled();
    expect(path).toBe('/local/path/model.bin');
  });

  it('resolveModel should strip file:// prefix from local paths', async () => {
    const path = await ModelRegistry.resolveModel('file:///local/path/model.bin');
    expect(mockModelStore.downloadFile).not.toHaveBeenCalled();
    expect(path).toBe('/local/path/model.bin');
  });

  it('resolveModel should forward progress callbacks from the native store', async () => {
    mockModelStore.downloadFile.mockImplementationOnce(
      async (_url: string, _file: string, _headers: string, _sha256: string, onProgress: (p: number) => void) => {
        onProgress(0.25);
        onProgress(0.5);
        onProgress(1);
        return '/downloaded/model.bin';
      },
    );
    const seen: number[] = [];
    await ModelRegistry.resolveModel('https://example.com/model.bin', {
      onProgress: (p) => seen.push(p),
    });
    expect(seen).toEqual([0.25, 0.5, 1]);
  });

  it('resolveModel should propagate native download failures', async () => {
    mockModelStore.downloadFile.mockRejectedValueOnce(new Error('HTTP 404'));
    await expect(
      ModelRegistry.resolveModel('https://example.com/missing.bin'),
    ).rejects.toThrow('HTTP 404');
  });

  it('resolveModel should reject URLs with no filename component', async () => {
    await expect(
      ModelRegistry.resolveModel('https://example.com/models/'),
    ).rejects.toThrow('Invalid model URL');
    expect(mockModelStore.downloadFile).not.toHaveBeenCalled();
  });

  it('resolveModel should default headers to an empty JSON object', async () => {
    mockModelStore.downloadFile.mockResolvedValueOnce('/downloaded/model.bin');
    await ModelRegistry.resolveModel('https://example.com/model.bin');
    expect(mockModelStore.downloadFile).toHaveBeenCalledWith(
      'https://example.com/model.bin',
      'model.bin',
      '{}',
      '',
      expect.any(Function),
    );
  });

  it('resolveModel should strip query strings from the cached filename', async () => {
    mockModelStore.downloadFile.mockResolvedValueOnce('/downloaded/model.bin');
    await ModelRegistry.resolveModel('https://example.com/model.bin?token=abc&x=1');
    expect(mockModelStore.downloadFile.mock.calls[0][1]).toBe('model.bin');
  });

  const SHA256 = '03E7DA1EB1108B50DFFAA9BB52CC7BCBAD2EB0C66CA990267F480C1E545D2856';

  it('resolveModel should pass the expected sha256 to native in lower case', async () => {
    mockModelStore.downloadFile.mockResolvedValueOnce('/downloaded/model.bin');
    await ModelRegistry.resolveModel('https://example.com/model.bin', { sha256: SHA256 });
    expect(mockModelStore.downloadFile).toHaveBeenCalledWith(
      'https://example.com/model.bin',
      'model.bin',
      '{}',
      SHA256.toLowerCase(),
      expect.any(Function),
    );
  });

  it.each([
    ['too short', SHA256.slice(1)],
    ['not hex', 'g' + SHA256.slice(1)],
    ['empty', ''],
  ])('resolveModel should reject a sha256 that is %s before downloading', async (_label, sha256) => {
    const err = await ModelRegistry.resolveModel('https://example.com/model.bin', { sha256 }).catch((e) => e);
    expect(err.code).toBe('INVALID_INPUT');
    expect(err.message).toContain('Invalid sha256');
    expect(mockModelStore.downloadFile).not.toHaveBeenCalled();
  });

  it('resolveModel should not check a sha256 for a local path', async () => {
    const path = await ModelRegistry.resolveModel('/local/path/model.bin', { sha256: 'not-a-hash' });
    expect(path).toBe('/local/path/model.bin');
    expect(mockModelStore.downloadFile).not.toHaveBeenCalled();
  });

  it('resolveModel should report a sha256 mismatch as DOWNLOAD_FAILED', async () => {
    mockModelStore.downloadFile.mockRejectedValueOnce(
      new Error(`java.lang.RuntimeException: Download failed: sha256 mismatch: expected ${SHA256.toLowerCase()}, got ${'0'.repeat(64)}`),
    );
    const err = await ModelRegistry.resolveModel('https://example.com/model.bin', { sha256: SHA256 }).catch((e) => e);
    expect(err.code).toBe('DOWNLOAD_FAILED');
    expect(err.message).toContain(`sha256 mismatch: expected ${SHA256.toLowerCase()}, got ${'0'.repeat(64)}`);
  });
});
