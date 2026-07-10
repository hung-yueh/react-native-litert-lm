/**
 * Memory tracking utilities for LiteRT-LM using real native memory metrics.
 *
 * Records real memory usage from OS-level APIs via `getMemoryUsage()`,
 * and stores snapshots in a native-backed ArrayBuffer allocated via
 * `NitroModules.createNativeArrayBuffer()` (v0.35+) for zero-copy interop.
 *
 * @example
 * ```typescript
 * import { createMemoryTracker } from 'react-native-litert-lm';
 *
 * const tracker = createMemoryTracker(100);
 *
 * // Record a real snapshot (typically called internally after inference)
 * tracker.record({
 *   timestamp: Date.now(),
 *   nativeHeapBytes: usage.nativeHeapBytes,
 *   residentBytes: usage.residentBytes,
 *   availableMemoryBytes: usage.availableMemoryBytes,
 * });
 *
 * console.log(`Peak RSS: ${tracker.getPeakMemory()} bytes`);
 * ```
 */

import { NitroModules } from "react-native-nitro-modules";

/**
 * A single memory usage snapshot with real data from OS APIs.
 */
export interface MemorySnapshot {
  /** Unix timestamp in milliseconds */
  timestamp: number;
  /** Native heap allocated bytes (Debug.getNativeHeapAllocatedSize on Android, task_info on iOS) */
  nativeHeapBytes: number;
  /** Process resident set size (RSS) in bytes */
  residentBytes: number;
  /** Available system memory in bytes */
  availableMemoryBytes: number;
}

/** Number of Float64 fields per snapshot */
const FIELDS_PER_SNAPSHOT = 4;
/** Bytes per Float64 value */
const BYTES_PER_FIELD = Float64Array.BYTES_PER_ELEMENT; // 8

/**
 * Memory tracker that stores snapshots in a native-backed ArrayBuffer.
 *
 * Uses `NitroModules.createNativeArrayBuffer()` to allocate the backing
 * buffer in native (C++) memory, ensuring zero-copy interop with native
 * methods and keeping memory tracking data off the JS heap.
 */
export interface MemoryTracker {
  /**
   * Record a new memory snapshot.
   * The tracker is a ring buffer: when full, the oldest snapshot is
   * overwritten so long-running sessions keep the most recent history.
   * @param snapshot The memory usage data to record
   * @returns always true (kept for backward compatibility)
   */
  record(snapshot: MemorySnapshot): boolean;

  /**
   * Total snapshots ever recorded (may exceed capacity once wrapped).
   */
  getTotalRecorded(): number;

  /**
   * Get all recorded snapshots as structured objects.
   */
  getSnapshots(): MemorySnapshot[];

  /**
   * Get the number of recorded snapshots.
   */
  getSnapshotCount(): number;

  /**
   * Get the maximum number of snapshots this tracker can hold.
   */
  getCapacity(): number;

  /**
   * Get the peak resident set size across all snapshots.
   */
  getPeakMemory(): number;

  /**
   * Get the latest memory snapshot, or undefined if none recorded.
   */
  getLatestSnapshot(): MemorySnapshot | undefined;

  /**
   * Get the underlying native ArrayBuffer.
   * This buffer is allocated via `NitroModules.createNativeArrayBuffer()`
   * and lives in native memory, enabling zero-copy transfer to native methods.
   */
  getNativeBuffer(): ArrayBuffer;

  /**
   * Get the Float64Array view over the native buffer.
   */
  getView(): Float64Array;

  /**
   * Reset the tracker, clearing all recorded snapshots.
   * The native buffer is preserved (not reallocated).
   */
  reset(): void;

  /**
   * Get a summary of memory usage statistics.
   */
  getSummary(): MemoryTrackerSummary;
}

/**
 * Summary statistics from the memory tracker.
 */
export interface MemoryTrackerSummary {
  /** Number of snapshots recorded */
  snapshotCount: number;
  /** Peak resident set size in bytes */
  peakResidentBytes: number;
  /** Average resident set size in bytes */
  averageResidentBytes: number;
  /** Latest resident set size in bytes */
  currentResidentBytes: number;
  /** Peak native heap allocated in bytes */
  peakNativeHeapBytes: number;
  /** Latest native heap allocated in bytes */
  currentNativeHeapBytes: number;
  /** RSS delta from first to last snapshot in bytes */
  residentDeltaBytes: number;
  /** Size of the native tracking buffer itself in bytes */
  trackerBufferSizeBytes: number;
}

/**
 * Create a new memory tracker backed by a native ArrayBuffer.
 *
 * @param maxSnapshots Maximum number of snapshots to store (default: 256)
 * @returns A MemoryTracker instance
 */
export function createMemoryTracker(maxSnapshots: number = 256): MemoryTracker {
  const bufferSize = maxSnapshots * FIELDS_PER_SNAPSHOT * BYTES_PER_FIELD;

  // Use NitroModules.createNativeArrayBuffer for native-backed allocation.
  const nativeBuffer = NitroModules.createNativeArrayBuffer(bufferSize);
  const view = new Float64Array(nativeBuffer);

  // Ring buffer state: `totalRecorded` grows forever; the physical slot for
  // logical entry i (0 = oldest retained) is derived from it.
  let totalRecorded = 0;

  const count = () => Math.min(totalRecorded, maxSnapshots);
  /** Physical slot offset of logical (chronological) index i. */
  const offsetOf = (i: number): number => {
    const start = totalRecorded > maxSnapshots ? totalRecorded % maxSnapshots : 0;
    return ((start + i) % maxSnapshots) * FIELDS_PER_SNAPSHOT;
  };

  const snapshotAt = (i: number): MemorySnapshot => {
    const offset = offsetOf(i);
    return {
      timestamp: view[offset]!,
      nativeHeapBytes: view[offset + 1]!,
      residentBytes: view[offset + 2]!,
      availableMemoryBytes: view[offset + 3]!,
    };
  };

  return {
    record(snapshot: MemorySnapshot): boolean {
      const offset = (totalRecorded % maxSnapshots) * FIELDS_PER_SNAPSHOT;
      view[offset] = snapshot.timestamp;
      view[offset + 1] = snapshot.nativeHeapBytes;
      view[offset + 2] = snapshot.residentBytes;
      view[offset + 3] = snapshot.availableMemoryBytes;
      totalRecorded++;
      return true;
    },

    getSnapshots(): MemorySnapshot[] {
      const n = count();
      const snapshots: MemorySnapshot[] = [];
      for (let i = 0; i < n; i++) {
        snapshots.push(snapshotAt(i));
      }
      return snapshots;
    },

    getSnapshotCount(): number {
      return count();
    },

    getTotalRecorded(): number {
      return totalRecorded;
    },

    getCapacity(): number {
      return maxSnapshots;
    },

    getPeakMemory(): number {
      const n = count();
      let peak = 0;
      for (let i = 0; i < n; i++) {
        const rss = view[offsetOf(i) + 2]!;
        if (rss > peak) {
          peak = rss;
        }
      }
      return peak;
    },

    getLatestSnapshot(): MemorySnapshot | undefined {
      const n = count();
      if (n === 0) return undefined;
      return snapshotAt(n - 1);
    },

    getNativeBuffer(): ArrayBuffer {
      return nativeBuffer;
    },

    getView(): Float64Array {
      return view;
    },

    reset(): void {
      view.fill(0);
      totalRecorded = 0;
    },

    getSummary(): MemoryTrackerSummary {
      const n = count();
      let peakRss = 0;
      let peakHeap = 0;
      let sumRss = 0;
      let firstRss = 0;
      let lastRss = 0;
      let lastHeap = 0;

      for (let i = 0; i < n; i++) {
        const offset = offsetOf(i);
        const heap = view[offset + 1]!;
        const rss = view[offset + 2]!;

        if (rss > peakRss) peakRss = rss;
        if (heap > peakHeap) peakHeap = heap;
        sumRss += rss;
        if (i === 0) firstRss = rss;
        if (i === n - 1) {
          lastRss = rss;
          lastHeap = heap;
        }
      }

      return {
        snapshotCount: n,
        peakResidentBytes: peakRss,
        averageResidentBytes: n > 0 ? sumRss / n : 0,
        currentResidentBytes: lastRss,
        peakNativeHeapBytes: peakHeap,
        currentNativeHeapBytes: lastHeap,
        residentDeltaBytes: lastRss - firstRss,
        trackerBufferSizeBytes: bufferSize,
      };
    },
  };
}

/**
 * Create a native ArrayBuffer for efficient data transfer.
 *
 * A convenience wrapper around `NitroModules.createNativeArrayBuffer()`.
 *
 * @param size Size in bytes
 * @returns A native-backed ArrayBuffer
 */
export function createNativeBuffer(size: number): ArrayBuffer {
  return NitroModules.createNativeArrayBuffer(size);
}
