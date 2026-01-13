# Matrix Tiling and Cache Behavior Visualizer - Specification

## Overview

An educational web-based tool that visualizes how iteration orders, tiling strategies, and data layouts affect cache behavior during matrix multiplication.

**Target Audience**: Students and educators learning about cache locality and loop optimization.

---

## Current Feature Set (v1.0 - Complete)

### Operation: Matrix Multiplication
- Computation: `C[i][j] += A[i][k] * B[k][j]`
- Fixed 12x12 matrices (1,728 iterations, 5,184 memory accesses)

### Configurable Parameters
| Parameter | Options |
|-----------|---------|
| Loop Order | ijk, ikj, jik, jki, kij, kji |
| Tiling | Off / On (2x2, 4x4, 6x6) |
| Data Layout | Row-major / Column-major (per matrix) |
| Cache Size | User-defined bytes (default 256) |

### Cache Model
- **Line size**: 64 bytes (16 elements)
- **Eviction**: LRU (Least Recently Used)
- **Element size**: 4 bytes

### Playback
- Play/Pause, Step Forward/Backward, Reset
- Speed control (1x-100x logarithmic)
- Jump to iteration (replays from start)

---

## Algorithmic Reference

### Memory Address Calculation
```
Row-major:    address = base + (row * 12 + col) * 4
Column-major: address = base + (col * 12 + row) * 4

Base addresses: A=0, B=576, C=1152
```

### Cache Behavior
- Access checks if cache line is present
- Hit: move line to MRU position
- Miss: load line, evict LRU if full

### Expected Behaviors (for verification)
| Configuration | Expected Result |
|---------------|-----------------|
| `ijk` row-major | Poor B locality (column access pattern) |
| `ikj` row-major | Good A and B locality |
| `kij` row-major | Good B and C locality |
| Tiling (large cache) | Improved hit rates |
| Tiling (tiny cache) | May hurt due to tile-switching overhead |
| Column-major B with `ijk` | Improves B locality |

---

## Potential Future Features

These are NOT currently implemented. See CLAUDE.md for architectural guidance on adding them.

### Partial Tiling
Tile some dimensions but not others (e.g., tile i and j, not k).
- Requires per-dimension tiling configuration
- Useful for demonstrating when tiling helps vs hurts

### Alternative Operations
- **2D Convolution**: 7 loop dimensions, different access patterns
- **Transpose**: Simple 2-loop operation
- **Vector operations**: 1D examples

### Enhanced Visualization
- Side-by-side comparison mode
- Cache associativity modeling (set-associative)
- Explanatory text generation

### Configuration
- Variable matrix sizes
- Custom loop orders
- Export/save results

---

## Technical Notes

### Constants
```javascript
MATRIX_SIZE = 12
ELEMENT_SIZE = 4 bytes
CACHE_LINE_SIZE = 64 bytes
TOTAL_ITERATIONS = 1728
```

### Iteration Count Formula
- Non-tiled: `size^3` iterations
- Tiled: same total, different ordering

### File Structure
```
index.html   - UI structure
styles.css   - Styling
app.js       - All logic
CLAUDE.md    - Development guidance
```
