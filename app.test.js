/**
 * Unit Tests for Cache Visualizer
 * Run with: npm test
 */

const { describe, it } = require('node:test');
const assert = require('node:assert');
const {
    createMatmulOperation,
    generateIterations,
    generateTiledIterations,
    CacheSimulator
} = require('./app.js');

// =============================================================================
// Operation Definition
// =============================================================================

describe('Operation Definition', () => {
    it('creates correct structure', () => {
        const op = createMatmulOperation(12, 4);
        assert.strictEqual(op.name, 'matmul');
        assert.strictEqual(op.size, 12);
        assert.strictEqual(op.elementSize, 4);
        assert.strictEqual(op.tensors.length, 3);
        assert.strictEqual(op.loopDims.length, 3);
    });

    it('has correct tensor names', () => {
        const op = createMatmulOperation(12, 4);
        const names = op.tensors.map(t => t.name);
        assert.deepStrictEqual(names, ['A', 'B', 'C']);
    });

    it('has non-overlapping tensor addresses', () => {
        const op = createMatmulOperation(12, 4);
        const [A, B, C] = op.tensors;
        assert.strictEqual(A.baseAddress, 0);
        assert.strictEqual(B.baseAddress, 12 * 12 * 4);
        assert.strictEqual(C.baseAddress, 2 * 12 * 12 * 4);
    });

    it('computes total iterations as size^3', () => {
        const op = createMatmulOperation(12, 4);
        assert.strictEqual(op.getTotalIterations(), 12 * 12 * 12);
    });

    it('maps tensor indices correctly', () => {
        const op = createMatmulOperation(12, 4);
        const iter = { i: 3, j: 5, k: 7 };

        // A[i][k], B[k][j], C[i][j]
        assert.deepStrictEqual(op.tensors[0].getIndices(iter), { row: 3, col: 7 });
        assert.deepStrictEqual(op.tensors[1].getIndices(iter), { row: 7, col: 5 });
        assert.deepStrictEqual(op.tensors[2].getIndices(iter), { row: 3, col: 5 });
    });
});

// =============================================================================
// Iteration Generator (Non-tiled)
// =============================================================================

describe('Iteration Generator (Non-tiled)', () => {
    const op = createMatmulOperation(4, 4);

    it('generates correct count', () => {
        const iters = generateIterations(op, 'ijk');
        assert.strictEqual(iters.length, 64);
    });

    it('ijk: i outermost, k innermost', () => {
        const iters = generateIterations(op, 'ijk');
        // First 16 iterations have i=0
        for (let n = 0; n < 16; n++) {
            assert.strictEqual(iters[n].i, 0);
        }
        // k cycles 0,1,2,3
        assert.strictEqual(iters[0].k, 0);
        assert.strictEqual(iters[1].k, 1);
        assert.strictEqual(iters[2].k, 2);
        assert.strictEqual(iters[3].k, 3);
        assert.strictEqual(iters[4].k, 0);
    });

    it('kji: k outermost, i innermost', () => {
        const iters = generateIterations(op, 'kji');
        // First 16 iterations have k=0
        for (let n = 0; n < 16; n++) {
            assert.strictEqual(iters[n].k, 0);
        }
        // i cycles 0,1,2,3
        assert.strictEqual(iters[0].i, 0);
        assert.strictEqual(iters[1].i, 1);
        assert.strictEqual(iters[2].i, 2);
        assert.strictEqual(iters[3].i, 3);
    });

    it('stays within bounds', () => {
        const iters = generateIterations(op, 'ijk');
        for (const iter of iters) {
            assert.ok(iter.i >= 0 && iter.i < 4);
            assert.ok(iter.j >= 0 && iter.j < 4);
            assert.ok(iter.k >= 0 && iter.k < 4);
        }
    });

    it('covers all combinations exactly once', () => {
        const iters = generateIterations(op, 'ijk');
        const seen = new Set(iters.map(it => `${it.i},${it.j},${it.k}`));
        assert.strictEqual(seen.size, 64);
    });
});

// =============================================================================
// Iteration Generator (Tiled)
// =============================================================================

describe('Iteration Generator (Tiled)', () => {
    const op = createMatmulOperation(4, 4);

    it('generates same count as non-tiled', () => {
        const iters = generateTiledIterations(op, 'ijk', 2);
        assert.strictEqual(iters.length, 64);
    });

    it('tile indices are multiples of tile size', () => {
        const iters = generateTiledIterations(op, 'ijk', 2);
        for (const iter of iters) {
            assert.strictEqual(iter.ti % 2, 0);
            assert.strictEqual(iter.tj % 2, 0);
            assert.strictEqual(iter.tk % 2, 0);
        }
    });

    it('element indices stay within tile', () => {
        const iters = generateTiledIterations(op, 'ijk', 2);
        for (const iter of iters) {
            assert.ok(iter.i >= iter.ti && iter.i < iter.ti + 2);
            assert.ok(iter.j >= iter.tj && iter.j < iter.tj + 2);
            assert.ok(iter.k >= iter.tk && iter.k < iter.tk + 2);
        }
    });

    it('processes first tile first', () => {
        const iters = generateTiledIterations(op, 'ijk', 2);
        // First 8 iterations in tile (0,0,0)
        for (let n = 0; n < 8; n++) {
            assert.strictEqual(iters[n].ti, 0);
            assert.strictEqual(iters[n].tj, 0);
            assert.strictEqual(iters[n].tk, 0);
        }
    });

    it('covers all combinations exactly once', () => {
        const iters = generateTiledIterations(op, 'ijk', 2);
        const seen = new Set(iters.map(it => `${it.i},${it.j},${it.k}`));
        assert.strictEqual(seen.size, 64);
    });
});

// =============================================================================
// Cache Simulator
// =============================================================================

describe('Cache Simulator', () => {
    it('initializes correctly', () => {
        const cache = new CacheSimulator(64, 16);
        assert.strictEqual(cache.maxLines, 4);
        assert.strictEqual(cache.lines.length, 0);
    });

    it('first access is a miss', () => {
        const cache = new CacheSimulator(64, 16);
        assert.strictEqual(cache.access(0), false);
        assert.strictEqual(cache.misses, 1);
    });

    it('same cache line is a hit', () => {
        const cache = new CacheSimulator(64, 16);
        cache.access(0);
        assert.strictEqual(cache.access(4), true);  // Same line (0-15)
        assert.strictEqual(cache.hits, 1);
    });

    it('different cache line is a miss', () => {
        const cache = new CacheSimulator(64, 16);
        cache.access(0);
        assert.strictEqual(cache.access(16), false);  // Different line
        assert.strictEqual(cache.misses, 2);
    });

    it('evicts LRU when full', () => {
        const cache = new CacheSimulator(32, 16);  // 2 lines
        cache.access(0);
        cache.access(16);
        cache.access(32);  // Evicts line 0

        assert.strictEqual(cache.isAddressCached(0), false);
        assert.strictEqual(cache.isAddressCached(16), true);
        assert.strictEqual(cache.isAddressCached(32), true);
    });

    it('access updates LRU order', () => {
        const cache = new CacheSimulator(32, 16);  // 2 lines
        cache.access(0);
        cache.access(16);
        cache.access(0);   // Touch line 0, now line 16 is LRU
        cache.access(32);  // Evicts line 16

        assert.strictEqual(cache.isAddressCached(0), true);
        assert.strictEqual(cache.isAddressCached(16), false);
    });

    it('aligns addresses to line boundary', () => {
        const cache = new CacheSimulator(64, 16);
        assert.strictEqual(cache.getLineAddress(0), 0);
        assert.strictEqual(cache.getLineAddress(15), 0);
        assert.strictEqual(cache.getLineAddress(16), 16);
        assert.strictEqual(cache.getLineAddress(31), 16);
    });

    it('reset clears state', () => {
        const cache = new CacheSimulator(64, 16);
        cache.access(0);
        cache.access(16);
        cache.reset();

        assert.strictEqual(cache.lines.length, 0);
        assert.strictEqual(cache.hits, 0);
        assert.strictEqual(cache.misses, 0);
    });

    it('snapshot/restore preserves state', () => {
        const cache = new CacheSimulator(64, 16);
        cache.access(0);
        cache.access(16);
        const snap = cache.snapshot();

        cache.access(32);
        cache.access(48);
        cache.restore(snap);

        assert.strictEqual(cache.lines.length, 2);
        assert.strictEqual(cache.isAddressCached(0), true);
        assert.strictEqual(cache.isAddressCached(32), false);
    });
});

// =============================================================================
// Cache Behavior (Educational scenarios)
// =============================================================================

describe('Cache Behavior', () => {
    it('4 lines sufficient for 3-tensor hits', () => {
        const cache = new CacheSimulator(64, 16);
        cache.access(0);     // Tensor A
        cache.access(576);   // Tensor B
        cache.access(1152);  // Tensor C

        // All still cached
        assert.strictEqual(cache.isAddressCached(0), true);
        assert.strictEqual(cache.isAddressCached(576), true);
        assert.strictEqual(cache.isAddressCached(1152), true);
    });

    it('spatial locality within cache line', () => {
        const cache = new CacheSimulator(256, 32);  // 8 elements per line
        cache.access(0);  // Load line with elements 0-7

        // Elements 1-7 should hit
        for (let i = 1; i < 8; i++) {
            assert.strictEqual(cache.access(i * 4), true, `element ${i}`);
        }
        // Element 8 is different line
        assert.strictEqual(cache.access(32), false);
    });
});
