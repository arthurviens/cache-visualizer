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
    CacheSimulator,
    getLinearIndex,
    getTensorAddress
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
// Linear Index Calculation
// =============================================================================

describe('Linear Index Calculation', () => {
    it('row-major: linearizes as row * size + col', () => {
        // For a 12x12 matrix in row-major layout
        assert.strictEqual(getLinearIndex(0, 0, 'row'), 0);
        assert.strictEqual(getLinearIndex(0, 5, 'row'), 5);
        assert.strictEqual(getLinearIndex(1, 0, 'row'), 12);
        assert.strictEqual(getLinearIndex(2, 3, 'row'), 27);  // 2*12 + 3
        assert.strictEqual(getLinearIndex(11, 11, 'row'), 143);  // last element
    });

    it('col-major: linearizes as col * size + row', () => {
        // For a 12x12 matrix in column-major layout
        assert.strictEqual(getLinearIndex(0, 0, 'col'), 0);
        assert.strictEqual(getLinearIndex(5, 0, 'col'), 5);
        assert.strictEqual(getLinearIndex(0, 1, 'col'), 12);
        assert.strictEqual(getLinearIndex(3, 2, 'col'), 27);  // 2*12 + 3
        assert.strictEqual(getLinearIndex(11, 11, 'col'), 143);  // last element
    });

    it('row-major: adjacent columns are contiguous', () => {
        // Elements in the same row should be adjacent in memory
        const row = 5;
        for (let col = 0; col < 11; col++) {
            const curr = getLinearIndex(row, col, 'row');
            const next = getLinearIndex(row, col + 1, 'row');
            assert.strictEqual(next - curr, 1, `col ${col} to ${col + 1}`);
        }
    });

    it('col-major: adjacent rows are contiguous', () => {
        // Elements in the same column should be adjacent in memory
        const col = 5;
        for (let row = 0; row < 11; row++) {
            const curr = getLinearIndex(row, col, 'col');
            const next = getLinearIndex(row + 1, col, 'col');
            assert.strictEqual(next - curr, 1, `row ${row} to ${row + 1}`);
        }
    });

    it('same element maps to different linear positions based on layout', () => {
        // Element [3][7] should have different linear index in row vs col major
        const rowMajor = getLinearIndex(3, 7, 'row');  // 3*12 + 7 = 43
        const colMajor = getLinearIndex(3, 7, 'col');  // 7*12 + 3 = 87
        assert.strictEqual(rowMajor, 43);
        assert.strictEqual(colMajor, 87);
        assert.notStrictEqual(rowMajor, colMajor);
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

// =============================================================================
// Tensor Address Calculation (for memory layout visualization)
// =============================================================================

describe('Tensor Address Calculation', () => {
    const op = createMatmulOperation(12, 4);
    const tensorA = op.tensors[0];  // A, baseAddress = 0
    const tensorB = op.tensors[1];  // B, baseAddress = 576
    const tensorC = op.tensors[2];  // C, baseAddress = 1152

    it('row-major: consecutive columns are adjacent in memory', () => {
        // For row-major, elements in the same row are contiguous
        const addr0 = getTensorAddress(tensorA, 0, 0, 'row');
        const addr1 = getTensorAddress(tensorA, 0, 1, 'row');
        const addr2 = getTensorAddress(tensorA, 0, 2, 'row');

        assert.strictEqual(addr1 - addr0, 4);  // 4 bytes apart
        assert.strictEqual(addr2 - addr1, 4);
    });

    it('col-major: consecutive rows are adjacent in memory', () => {
        // For col-major, elements in the same column are contiguous
        const addr0 = getTensorAddress(tensorA, 0, 0, 'col');
        const addr1 = getTensorAddress(tensorA, 1, 0, 'col');
        const addr2 = getTensorAddress(tensorA, 2, 0, 'col');

        assert.strictEqual(addr1 - addr0, 4);  // 4 bytes apart
        assert.strictEqual(addr2 - addr1, 4);
    });

    it('different tensors have non-overlapping address ranges', () => {
        // Get address range for each tensor (all elements)
        const aMax = getTensorAddress(tensorA, 11, 11, 'row');
        const bMin = getTensorAddress(tensorB, 0, 0, 'row');
        const bMax = getTensorAddress(tensorB, 11, 11, 'row');
        const cMin = getTensorAddress(tensorC, 0, 0, 'row');

        assert.ok(aMax < bMin, 'A should end before B starts');
        assert.ok(bMax < cMin, 'B should end before C starts');
    });

    it('cache can track tensor elements correctly', () => {
        // This test verifies the integration between getTensorAddress and CacheSimulator
        // which is the core of isElementInCache2D functionality
        const cache = new CacheSimulator(256, 64);  // 16 elements per line

        // Access element A[0][0] (address 0)
        const addrA00 = getTensorAddress(tensorA, 0, 0, 'row');
        cache.access(addrA00);

        // Element A[0][5] should be in same cache line (row-major, elements 0-15 in line)
        const addrA05 = getTensorAddress(tensorA, 0, 5, 'row');
        assert.ok(cache.isAddressCached(addrA05), 'A[0][5] should be cached');

        // Element A[1][0] is in a different cache line (element 12 in linear order)
        // Actually, A[1][0] is at linear index 12, which is 48 bytes from start
        // Cache line is 64 bytes, so 0-63 are in one line
        // A[1][0] is at offset 12*4 = 48, still in first line
        const addrA10 = getTensorAddress(tensorA, 1, 0, 'row');
        assert.ok(cache.isAddressCached(addrA10), 'A[1][0] should be cached (same 64B line)');

        // Element B[0][0] is NOT cached (different tensor, different address)
        const addrB00 = getTensorAddress(tensorB, 0, 0, 'row');
        assert.ok(!cache.isAddressCached(addrB00), 'B[0][0] should not be cached');
    });
});
