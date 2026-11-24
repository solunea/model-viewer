/* @license Copyright 2026 Solunea. Licensed under the Apache License,
 * Version 2.0. */

// Fixed, centered patterns in shadow-map space. Using the same offsets for
// every pixel avoids the grain and shimmer of randomly rotating the disk.
export const pcssBlockerOffsets = [
  [0, 0],
  [0.25, 0.14], [-0.25, -0.14],
  [-0.31, 0.50], [0.31, -0.50],
  [0.68, 0.52], [-0.68, -0.52],
  [-0.43, 0.82], [0.43, -0.82],
] as const;

export const pcssDepthEpsilon = 1e-6;
