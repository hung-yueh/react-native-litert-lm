import { createMemoryTracker, createNativeBuffer } from '../memoryTracker';
import { NitroModules } from 'react-native-nitro-modules';

describe('MemoryTracker Unit Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should allocate correct native-backed ArrayBuffer size on initialization', () => {
    const tracker = createMemoryTracker(10);
    expect(NitroModules.createNativeArrayBuffer).toHaveBeenCalledWith(10 * 4 * 8); // 10 snapshots * 4 fields * 8 bytes/Float64
    expect(tracker.getCapacity()).toBe(10);
    expect(tracker.getSnapshotCount()).toBe(0);
  });

  it('should record snapshots correctly and retrieve them', () => {
    const tracker = createMemoryTracker(5);
    const snapshot1 = {
      timestamp: 1000,
      nativeHeapBytes: 100,
      residentBytes: 200,
      availableMemoryBytes: 500,
    };
    
    expect(tracker.record(snapshot1)).toBe(true);
    expect(tracker.getSnapshotCount()).toBe(1);
    expect(tracker.getLatestSnapshot()).toEqual(snapshot1);
    
    const snapshots = tracker.getSnapshots();
    expect(snapshots).toHaveLength(1);
    expect(snapshots[0]).toEqual(snapshot1);
  });

  it('should wrap around as a ring buffer when capacity is reached, keeping the newest snapshots', () => {
    const tracker = createMemoryTracker(2);

    expect(tracker.record({ timestamp: 1, nativeHeapBytes: 10, residentBytes: 20, availableMemoryBytes: 50 })).toBe(true);
    expect(tracker.record({ timestamp: 2, nativeHeapBytes: 20, residentBytes: 30, availableMemoryBytes: 40 })).toBe(true);
    // Ring buffer: third record overwrites the oldest instead of being rejected
    expect(tracker.record({ timestamp: 3, nativeHeapBytes: 30, residentBytes: 40, availableMemoryBytes: 30 })).toBe(true);

    expect(tracker.getSnapshotCount()).toBe(2);
    expect(tracker.getTotalRecorded()).toBe(3);

    // Oldest (timestamp 1) evicted; chronological order preserved
    const snapshots = tracker.getSnapshots();
    expect(snapshots.map((s) => s.timestamp)).toEqual([2, 3]);
    expect(tracker.getLatestSnapshot()?.timestamp).toBe(3);
    // Peak scans only retained snapshots (40 from timestamp 3)
    expect(tracker.getPeakMemory()).toBe(40);
  });

  it('should calculate correct peak resident memory size', () => {
    const tracker = createMemoryTracker(5);
    tracker.record({ timestamp: 1, nativeHeapBytes: 100, residentBytes: 150, availableMemoryBytes: 1000 });
    tracker.record({ timestamp: 2, nativeHeapBytes: 120, residentBytes: 300, availableMemoryBytes: 1000 });
    tracker.record({ timestamp: 3, nativeHeapBytes: 110, residentBytes: 200, availableMemoryBytes: 1000 });

    expect(tracker.getPeakMemory()).toBe(300);
  });

  it('should calculate accurate memory summary statistics', () => {
    const tracker = createMemoryTracker(5);
    tracker.record({ timestamp: 1, nativeHeapBytes: 50, residentBytes: 100, availableMemoryBytes: 1000 });
    tracker.record({ timestamp: 2, nativeHeapBytes: 150, residentBytes: 300, availableMemoryBytes: 800 });
    tracker.record({ timestamp: 3, nativeHeapBytes: 100, residentBytes: 200, availableMemoryBytes: 900 });

    const summary = tracker.getSummary();
    expect(summary.snapshotCount).toBe(3);
    expect(summary.peakResidentBytes).toBe(300);
    expect(summary.averageResidentBytes).toBe(200); // (100 + 300 + 200) / 3
    expect(summary.currentResidentBytes).toBe(200);
    expect(summary.peakNativeHeapBytes).toBe(150);
    expect(summary.currentNativeHeapBytes).toBe(100);
    expect(summary.residentDeltaBytes).toBe(100); // currentRss(200) - firstRss(100)
    expect(summary.trackerBufferSizeBytes).toBe(5 * 4 * 8);
  });

  it('should preserve buffer but reset internal state when reset() is called', () => {
    const tracker = createMemoryTracker(5);
    tracker.record({ timestamp: 1, nativeHeapBytes: 50, residentBytes: 100, availableMemoryBytes: 1000 });
    
    expect(tracker.getSnapshotCount()).toBe(1);
    tracker.reset();
    expect(tracker.getSnapshotCount()).toBe(0);
    expect(tracker.getLatestSnapshot()).toBeUndefined();
    expect(tracker.getSnapshots()).toEqual([]);
  });

  it('should allow standalone native ArrayBuffer allocation via createNativeBuffer', () => {
    const size = 128;
    const buffer = createNativeBuffer(size);
    expect(NitroModules.createNativeArrayBuffer).toHaveBeenCalledWith(size);
    expect(buffer.byteLength).toBe(size);
  });
});
