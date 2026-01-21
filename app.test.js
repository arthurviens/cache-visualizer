/**
 * Unit Tests for Tensor Operations Cache Visualizer
 *
 * Tests cover:
 * - Operation definitions (matmul, conv2d)
 * - Iteration generation (non-tiled and tiled)
 * - Partial tiling for convolution
 * - Cache simulation (LRU eviction)
 * - Tensor address calculation
 *
 * Run with: npm test
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';
import {
    createMatmulOperation,
    createConv2dOperation,
    generateIterations,
    generateTiledIterations,
    CacheSimulator,
    getLinearIndex,
    getTensorAddress
} from './src/main.js';

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

    it('has tiling configuration for all dimensions', () => {
        const op = createMatmulOperation(12, 4);
        assert.deepStrictEqual(op.tileableDims, ['i', 'j', 'k']);
        assert.deepStrictEqual(op.tileSizes, [2, 4, 6]);
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

// =============================================================================
// Conv2D Operation Definition
// =============================================================================

describe('Conv2D Operation Definition', () => {
    it('creates correct structure with default config', () => {
        const op = createConv2dOperation();
        assert.strictEqual(op.name, 'conv2d');
        assert.strictEqual(op.tensors.length, 3);
        assert.strictEqual(op.loopDims.length, 6);
    });

    it('has correct tensor names', () => {
        const op = createConv2dOperation();
        const names = op.tensors.map(t => t.name);
        assert.deepStrictEqual(names, ['Input', 'Kernel', 'Output']);
    });

    it('has correct loop dimensions', () => {
        const op = createConv2dOperation();
        assert.deepStrictEqual(op.loopDims, ['c_out', 'h_out', 'w_out', 'c_in', 'k_h', 'k_w']);
    });

    it('computes correct output dimensions', () => {
        // Default: inputH=8, inputW=8, kernelH=3, kernelW=3
        // Output: (8-3+1) x (8-3+1) = 6x6
        const op = createConv2dOperation();
        assert.strictEqual(op.dimensions.outputH, 6);
        assert.strictEqual(op.dimensions.outputW, 6);
    });

    it('computes correct total iterations', () => {
        // c_out=4, h_out=6, w_out=6, c_in=4, k_h=3, k_w=3
        // Total: 4 * 6 * 6 * 4 * 3 * 3 = 5184
        const op = createConv2dOperation();
        assert.strictEqual(op.getTotalIterations(), 4 * 6 * 6 * 4 * 3 * 3);
    });

    it('has non-overlapping tensor addresses', () => {
        const op = createConv2dOperation();
        const [input, kernel, output] = op.tensors;

        // Input ends before Kernel starts
        const inputSize = 8 * 8 * 4;  // H * W * C_in
        assert.strictEqual(kernel.baseAddress, inputSize * 4);

        // Kernel ends before Output starts
        const kernelSize = 4 * 4 * 3 * 3;  // C_out * C_in * K_h * K_w
        assert.strictEqual(output.baseAddress, (inputSize + kernelSize) * 4);
    });

    it('allows custom configuration', () => {
        const op = createConv2dOperation({
            inputH: 16,
            inputW: 16,
            channels_in: 8,
            channels_out: 16,
            kernelH: 5,
            kernelW: 5
        });

        assert.strictEqual(op.dimensions.inputH, 16);
        assert.strictEqual(op.dimensions.inputW, 16);
        assert.strictEqual(op.dimensions.channels_in, 8);
        assert.strictEqual(op.dimensions.channels_out, 16);
        assert.strictEqual(op.dimensions.outputH, 12);  // 16 - 5 + 1
        assert.strictEqual(op.dimensions.outputW, 12);
    });

    it('has partial tiling configuration for spatial dimensions only', () => {
        const op = createConv2dOperation();
        // Only h_out and w_out are tiled (not c_out, c_in, k_h, k_w)
        assert.deepStrictEqual(op.tileableDims, ['h_out', 'w_out']);
        assert.deepStrictEqual(op.tileSizes, [2, 4]);
    });
});

// =============================================================================
// Conv2D Iteration Generator
// =============================================================================

describe('Conv2D Iteration Generator', () => {
    const op = createConv2dOperation({
        inputH: 4,
        inputW: 4,
        channels_in: 2,
        channels_out: 2,
        kernelH: 2,
        kernelW: 2
    });
    // Output: 3x3, total iterations: 2 * 3 * 3 * 2 * 2 * 2 = 144

    it('generates correct count', () => {
        const iters = generateIterations(op, 'c_out,h_out,w_out,c_in,k_h,k_w');
        assert.strictEqual(iters.length, op.getTotalIterations());
    });

    it('stays within bounds', () => {
        const iters = generateIterations(op, 'c_out,h_out,w_out,c_in,k_h,k_w');
        for (const iter of iters) {
            assert.ok(iter.c_out >= 0 && iter.c_out < 2, `c_out=${iter.c_out}`);
            assert.ok(iter.h_out >= 0 && iter.h_out < 3, `h_out=${iter.h_out}`);
            assert.ok(iter.w_out >= 0 && iter.w_out < 3, `w_out=${iter.w_out}`);
            assert.ok(iter.c_in >= 0 && iter.c_in < 2, `c_in=${iter.c_in}`);
            assert.ok(iter.k_h >= 0 && iter.k_h < 2, `k_h=${iter.k_h}`);
            assert.ok(iter.k_w >= 0 && iter.k_w < 2, `k_w=${iter.k_w}`);
        }
    });

    it('covers all combinations exactly once', () => {
        const iters = generateIterations(op, 'c_out,h_out,w_out,c_in,k_h,k_w');
        const seen = new Set(iters.map(it =>
            `${it.c_out},${it.h_out},${it.w_out},${it.c_in},${it.k_h},${it.k_w}`
        ));
        assert.strictEqual(seen.size, op.getTotalIterations());
    });

    it('outermost loop varies slowest', () => {
        const iters = generateIterations(op, 'c_out,h_out,w_out,c_in,k_h,k_w');
        // First half should have c_out=0
        const halfCount = op.getTotalIterations() / 2;
        for (let i = 0; i < halfCount; i++) {
            assert.strictEqual(iters[i].c_out, 0, `iter ${i}`);
        }
        // Second half should have c_out=1
        for (let i = halfCount; i < op.getTotalIterations(); i++) {
            assert.strictEqual(iters[i].c_out, 1, `iter ${i}`);
        }
    });

    it('tiled iteration generates correct count', () => {
        const iters = generateTiledIterations(op, 'c_out,h_out,w_out,c_in,k_h,k_w', 2);
        assert.strictEqual(iters.length, op.getTotalIterations());
    });

    it('tiled iteration covers all combinations exactly once', () => {
        const iters = generateTiledIterations(op, 'c_out,h_out,w_out,c_in,k_h,k_w', 2);
        const seen = new Set(iters.map(it =>
            `${it.c_out},${it.h_out},${it.w_out},${it.c_in},${it.k_h},${it.k_w}`
        ));
        assert.strictEqual(seen.size, op.getTotalIterations());
    });

    it('tiled iteration has correct tile indices for tileable dims only', () => {
        // Conv2d only tiles h_out and w_out (not c_out, c_in, k_h, k_w)
        const iters = generateTiledIterations(op, 'c_out,h_out,w_out,c_in,k_h,k_w', 2);
        for (const iter of iters) {
            // Only h_out and w_out should have tile indices
            assert.strictEqual(iter.th_out % 2, 0, `th_out=${iter.th_out}`);
            assert.strictEqual(iter.tw_out % 2, 0, `tw_out=${iter.tw_out}`);

            // Element indices for tiled dims should be within tile
            assert.ok(iter.h_out >= iter.th_out && iter.h_out < iter.th_out + 2,
                `h_out=${iter.h_out} not in tile [${iter.th_out}, ${iter.th_out + 2})`);
            assert.ok(iter.w_out >= iter.tw_out && iter.w_out < iter.tw_out + 2,
                `w_out=${iter.w_out} not in tile [${iter.tw_out}, ${iter.tw_out + 2})`);

            // Non-tiled dimensions should NOT have tile indices
            assert.strictEqual(iter.tc_out, undefined, 'c_out should not be tiled');
            assert.strictEqual(iter.tc_in, undefined, 'c_in should not be tiled');
            assert.strictEqual(iter.tk_h, undefined, 'k_h should not be tiled');
            assert.strictEqual(iter.tk_w, undefined, 'k_w should not be tiled');

            // Non-tiled dimensions should still have their regular indices
            assert.ok(iter.c_out >= 0 && iter.c_out < op.loopBounds.c_out);
            assert.ok(iter.c_in >= 0 && iter.c_in < op.loopBounds.c_in);
            assert.ok(iter.k_h >= 0 && iter.k_h < op.loopBounds.k_h);
            assert.ok(iter.k_w >= 0 && iter.k_w < op.loopBounds.k_w);
        }
    });

    it('partial tiling: c_out varies before tile loops', () => {
        // Loop order: c_out, h_out, w_out, c_in, k_h, k_w
        // With partial tiling (h_out, w_out only), structure should be:
        //   c_out (outer non-tiled) → th_out → tw_out → c_in, k_h, k_w (inner non-tiled) → h_out, w_out (element)
        const iters = generateTiledIterations(op, 'c_out,h_out,w_out,c_in,k_h,k_w', 2);

        // Find where c_out changes from 0 to 1
        // It should only change when we've exhausted all tile/element combinations for c_out=0
        let firstC1Index = iters.findIndex(it => it.c_out === 1);

        // All iterations before firstC1Index should have c_out=0
        for (let i = 0; i < firstC1Index; i++) {
            assert.strictEqual(iters[i].c_out, 0, `iter ${i} should have c_out=0`);
        }

        // The first iteration with c_out=1 should restart tile loops from the beginning
        assert.strictEqual(iters[firstC1Index].th_out, 0, 'th_out should reset to 0 for c_out=1');
        assert.strictEqual(iters[firstC1Index].tw_out, 0, 'tw_out should reset to 0 for c_out=1');
    });
});

// =============================================================================
// Conv2D Linear Index Calculation
// =============================================================================

describe('Conv2D Linear Index Calculation', () => {
    const op = createConv2dOperation();
    const [input, kernel, output] = op.tensors;

    it('Input tensor: linear index calculation', () => {
        // Input[c_in][h][w] - linear: c_in * (H * W) + h * W + w
        const iter = { c_in: 1, h_out: 2, w_out: 3, k_h: 1, k_w: 0, c_out: 0 };
        // Input access: h = h_out + k_h = 3, w = w_out + k_w = 3
        const linearIdx = input.getLinearIndex(iter);
        // Expected: 1 * (8 * 8) + 3 * 8 + 3 = 64 + 24 + 3 = 91
        assert.strictEqual(linearIdx, 91);
    });

    it('Kernel tensor: linear index calculation', () => {
        // Kernel[c_out][c_in][k_h][k_w]
        // Linear: c_out * (C_in * K_h * K_w) + c_in * (K_h * K_w) + k_h * K_w + k_w
        const iter = { c_out: 1, c_in: 2, k_h: 1, k_w: 2, h_out: 0, w_out: 0 };
        const linearIdx = kernel.getLinearIndex(iter);
        // Expected: 1 * (4 * 3 * 3) + 2 * (3 * 3) + 1 * 3 + 2 = 36 + 18 + 3 + 2 = 59
        assert.strictEqual(linearIdx, 59);
    });

    it('Output tensor: linear index calculation', () => {
        // Output[c_out][h_out][w_out]
        // Linear: c_out * (H_out * W_out) + h_out * W_out + w_out
        const iter = { c_out: 2, h_out: 3, w_out: 4, c_in: 0, k_h: 0, k_w: 0 };
        const linearIdx = output.getLinearIndex(iter);
        // Expected: 2 * (6 * 6) + 3 * 6 + 4 = 72 + 18 + 4 = 94
        assert.strictEqual(linearIdx, 94);
    });
});

// =============================================================================
// Tensor getTotalElements
// =============================================================================

describe('Tensor getTotalElements', () => {
    it('matmul tensors: 12x12 = 144 elements each', () => {
        const op = createMatmulOperation(12, 4);
        for (const tensor of op.tensors) {
            assert.strictEqual(tensor.getTotalElements(), 144, `${tensor.name} should have 144 elements`);
        }
    });

    it('conv2d Input tensor: 4 channels * 8 * 8 = 256 elements', () => {
        const op = createConv2dOperation();
        const input = op.tensors[0];
        assert.strictEqual(input.getTotalElements(), 256);
    });

    it('conv2d Kernel tensor: 4 * 4 * 3 * 3 = 144 elements', () => {
        const op = createConv2dOperation();
        const kernel = op.tensors[1];
        assert.strictEqual(kernel.getTotalElements(), 144);
    });

    it('conv2d Output tensor: 4 channels * 6 * 6 = 144 elements', () => {
        const op = createConv2dOperation();
        const output = op.tensors[2];
        assert.strictEqual(output.getTotalElements(), 144);
    });
});

// =============================================================================
// Tensor getCoordinatesFromLinear (inverse of getLinearIndex)
// =============================================================================

describe('Tensor getCoordinatesFromLinear', () => {
    describe('Matmul tensors (2D)', () => {
        const op = createMatmulOperation(12, 4);
        const [A, B, C] = op.tensors;
        const layout = 'row'; // Default layout for matmul

        it('Matrix A: inverse of getLinearIndex', () => {
            // Test several positions. A[i][k] accesses via i, k
            for (let row = 0; row < 12; row += 3) {
                for (let col = 0; col < 12; col += 4) {
                    const iter = { i: row, j: 0, k: col };
                    const linearIdx = A.getLinearIndex(iter, layout);
                    const coords = A.getCoordinatesFromLinear(linearIdx, layout);
                    assert.strictEqual(coords.row, row, `row mismatch at linear ${linearIdx}`);
                    assert.strictEqual(coords.col, col, `col mismatch at linear ${linearIdx}`);
                }
            }
        });

        it('Matrix B: inverse of getLinearIndex', () => {
            // B[k][j] accesses via k, j
            for (let row = 0; row < 12; row += 3) {
                for (let col = 0; col < 12; col += 4) {
                    const iter = { i: 0, j: col, k: row };
                    const linearIdx = B.getLinearIndex(iter, layout);
                    const coords = B.getCoordinatesFromLinear(linearIdx, layout);
                    assert.strictEqual(coords.row, row, `row mismatch at linear ${linearIdx}`);
                    assert.strictEqual(coords.col, col, `col mismatch at linear ${linearIdx}`);
                }
            }
        });

        it('Matrix C: inverse of getLinearIndex', () => {
            // C[i][j] accesses via i, j
            for (let row = 0; row < 12; row += 3) {
                for (let col = 0; col < 12; col += 4) {
                    const iter = { i: row, j: col, k: 0 };
                    const linearIdx = C.getLinearIndex(iter, layout);
                    const coords = C.getCoordinatesFromLinear(linearIdx, layout);
                    assert.strictEqual(coords.row, row, `row mismatch at linear ${linearIdx}`);
                    assert.strictEqual(coords.col, col, `col mismatch at linear ${linearIdx}`);
                }
            }
        });
    });

    describe('Conv2d tensors (3D/4D)', () => {
        const op = createConv2dOperation();
        const [input, kernel, output] = op.tensors;

        it('Input tensor (3D CHW): inverse of getLinearIndex', () => {
            // Test several positions across channels and spatial dims
            for (let c = 0; c < 4; c += 2) {
                for (let h = 0; h < 8; h += 3) {
                    for (let w = 0; w < 8; w += 3) {
                        const iter = { c_in: c, h_out: h, w_out: w, k_h: 0, k_w: 0, c_out: 0 };
                        const linearIdx = input.getLinearIndex(iter);
                        const coords = input.getCoordinatesFromLinear(linearIdx);
                        assert.strictEqual(coords.channel, c, `channel mismatch at linear ${linearIdx}`);
                        assert.strictEqual(coords.row, h, `row mismatch at linear ${linearIdx}`);
                        assert.strictEqual(coords.col, w, `col mismatch at linear ${linearIdx}`);
                    }
                }
            }
        });

        it('Kernel tensor (4D OIHW): inverse of getLinearIndex', () => {
            // Test several positions
            for (let c_out = 0; c_out < 4; c_out += 2) {
                for (let c_in = 0; c_in < 4; c_in += 2) {
                    for (let kh = 0; kh < 3; kh++) {
                        for (let kw = 0; kw < 3; kw++) {
                            const iter = { c_out, c_in, k_h: kh, k_w: kw, h_out: 0, w_out: 0 };
                            const linearIdx = kernel.getLinearIndex(iter);
                            const coords = kernel.getCoordinatesFromLinear(linearIdx);
                            assert.strictEqual(coords.c_out, c_out, `c_out mismatch at linear ${linearIdx}`);
                            assert.strictEqual(coords.c_in, c_in, `c_in mismatch at linear ${linearIdx}`);
                            assert.strictEqual(coords.row, kh, `row mismatch at linear ${linearIdx}`);
                            assert.strictEqual(coords.col, kw, `col mismatch at linear ${linearIdx}`);
                        }
                    }
                }
            }
        });

        it('Output tensor (3D CHW): inverse of getLinearIndex', () => {
            for (let c = 0; c < 4; c += 2) {
                for (let h = 0; h < 6; h += 2) {
                    for (let w = 0; w < 6; w += 2) {
                        const iter = { c_out: c, h_out: h, w_out: w, c_in: 0, k_h: 0, k_w: 0 };
                        const linearIdx = output.getLinearIndex(iter);
                        const coords = output.getCoordinatesFromLinear(linearIdx);
                        assert.strictEqual(coords.channel, c, `channel mismatch at linear ${linearIdx}`);
                        assert.strictEqual(coords.row, h, `row mismatch at linear ${linearIdx}`);
                        assert.strictEqual(coords.col, w, `col mismatch at linear ${linearIdx}`);
                    }
                }
            }
        });
    });

    it('exhaustive check: all linear indices map back correctly for matmul A', () => {
        const op = createMatmulOperation(12, 4);
        const A = op.tensors[0];
        const layout = 'row';
        const total = A.getTotalElements();
        for (let i = 0; i < total; i++) {
            const coords = A.getCoordinatesFromLinear(i, layout);
            // Reconstruct linear index from coords (row-major: row * cols + col)
            const reconstructed = coords.row * 12 + coords.col;
            assert.strictEqual(reconstructed, i, `linear index ${i} failed round-trip`);
        }
    });

    it('exhaustive check: all linear indices map back correctly for conv2d output', () => {
        const op = createConv2dOperation();
        const output = op.tensors[2];
        const total = output.getTotalElements();
        for (let i = 0; i < total; i++) {
            const coords = output.getCoordinatesFromLinear(i);
            // Reconstruct: channel * (6 * 6) + row * 6 + col
            const reconstructed = coords.channel * 36 + coords.row * 6 + coords.col;
            assert.strictEqual(reconstructed, i, `linear index ${i} failed round-trip`);
        }
    });
});
