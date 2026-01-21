/**
 * Tensor Operations Cache Behavior Visualizer
 *
 * An educational tool for visualizing how iteration orders, tiling strategies,
 * and data layouts affect cache behavior during tensor operations
 * (matrix multiplication and 2D convolution).
 */

// =============================================================================
// OPERATION ABSTRACTION
// =============================================================================
//
// An Operation defines:
// - name, displayName: Operation identifiers
// - tensors: Array of tensor definitions with access patterns
// - loopDims: Loop dimension names (e.g., ['i', 'j', 'k'] for matmul)
// - loopBounds: Bounds for each loop dimension
// - loopOrders: Available loop orderings
// - tileableDims: Which dimensions can be tiled
// - tileSizes: Available tile sizes
// - codeTemplate: Inner statement for code display
// - describeOp(iter): Human-readable description of current operation
// - getTotalIterations(): Total number of iterations
//
// Each tensor defines:
// - name, baseAddress, rows, cols (and channels for 3D/4D)
// - getIndices(iter): Map loop indices to tensor coordinates
// - getLinearIndex(iter, layout): For multi-dimensional tensors
// - layoutOptions: Available data layouts (optional)
//
// =============================================================================

/**
 * Creates the matrix multiplication operation.
 * C[i][j] += A[i][k] * B[k][j]
 *
 * @param {number} size - Matrix dimension (e.g., 12)
 * @param {number} elementSize - Bytes per element (e.g., 4)
 * @returns {Object} Operation definition
 */
function createMatmulOperation(size, elementSize) {
    const elementsPerMatrix = size * size;

    return {
        name: 'matmul',
        displayName: 'Matrix Multiplication',
        size: size,
        elementSize: elementSize,

        // Loop dimensions for this operation
        loopDims: ['i', 'j', 'k'],

        // Bounds for each loop dimension
        loopBounds: { i: size, j: size, k: size },

        // All valid loop orderings
        loopOrders: {
            'ijk': ['i', 'j', 'k'],
            'ikj': ['i', 'k', 'j'],
            'jik': ['j', 'i', 'k'],
            'jki': ['j', 'k', 'i'],
            'kij': ['k', 'i', 'j'],
            'kji': ['k', 'j', 'i']
        },

        // Tensor definitions
        // Each tensor has: name, baseAddress, accessIndices (which loop vars index it)
        tensors: [
            {
                name: 'A',
                baseAddress: 0,
                rows: size,
                cols: size,
                // A[i][k] - row is i, col is k
                getIndices: (iter) => ({ row: iter.i, col: iter.k }),
                getTotalElements: () => size * size,
                getLinearIndex: (iter, layout) => {
                    const { row, col } = { row: iter.i, col: iter.k };
                    return layout === 'col' ? col * size + row : row * size + col;
                },
                getCoordinatesFromLinear: (linearIdx, layout) => {
                    if (layout === 'col') {
                        return { row: linearIdx % size, col: Math.floor(linearIdx / size) };
                    }
                    return { row: Math.floor(linearIdx / size), col: linearIdx % size };
                }
            },
            {
                name: 'B',
                baseAddress: elementsPerMatrix * elementSize,
                rows: size,
                cols: size,
                // B[k][j] - row is k, col is j
                getIndices: (iter) => ({ row: iter.k, col: iter.j }),
                getTotalElements: () => size * size,
                getLinearIndex: (iter, layout) => {
                    const { row, col } = { row: iter.k, col: iter.j };
                    return layout === 'col' ? col * size + row : row * size + col;
                },
                getCoordinatesFromLinear: (linearIdx, layout) => {
                    if (layout === 'col') {
                        return { row: linearIdx % size, col: Math.floor(linearIdx / size) };
                    }
                    return { row: Math.floor(linearIdx / size), col: linearIdx % size };
                }
            },
            {
                name: 'C',
                baseAddress: 2 * elementsPerMatrix * elementSize,
                rows: size,
                cols: size,
                // C[i][j] - row is i, col is j
                getIndices: (iter) => ({ row: iter.i, col: iter.j }),
                getTotalElements: () => size * size,
                getLinearIndex: (iter, layout) => {
                    const { row, col } = { row: iter.i, col: iter.j };
                    return layout === 'col' ? col * size + row : row * size + col;
                },
                getCoordinatesFromLinear: (linearIdx, layout) => {
                    if (layout === 'col') {
                        return { row: linearIdx % size, col: Math.floor(linearIdx / size) };
                    }
                    return { row: Math.floor(linearIdx / size), col: linearIdx % size };
                }
            }
        ],

        // Inner statement for code display
        codeTemplate: 'C[i][j] += A[i][k] * B[k][j]',

        // Human-readable operation description
        describeOp: (iter) => `A[${iter.i}][${iter.k}] × B[${iter.k}][${iter.j}] → C[${iter.i}][${iter.j}]`,

        // Total iterations
        getTotalIterations: () => size * size * size,

        // Operators displayed between tensors in visualization (length = tensors.length - 1)
        tensorOperators: ['×', '='],

        // Tiling configuration
        tileableDims: ['i', 'j', 'k'],  // All dimensions can be tiled
        tileSizes: [2, 4, 6]            // Available tile sizes
    };
}

/**
 * Creates the 2D convolution operation.
 * Output[c_out][h_out][w_out] += Input[c_in][h_out+k_h][w_out+k_w] * Kernel[c_out][c_in][k_h][k_w]
 *
 * @param {Object} config - Configuration object
 * @param {number} config.inputH - Input height (default 8)
 * @param {number} config.inputW - Input width (default 8)
 * @param {number} config.channels_in - Input channels (default 4)
 * @param {number} config.channels_out - Output channels (default 4)
 * @param {number} config.kernelH - Kernel height (default 3)
 * @param {number} config.kernelW - Kernel width (default 3)
 * @param {number} config.elementSize - Bytes per element (default 4)
 * @returns {Object} Operation definition
 */
function createConv2dOperation(config = {}) {
    const {
        inputH = 8,
        inputW = 8,
        channels_in = 4,
        channels_out = 4,
        kernelH = 3,
        kernelW = 3,
        elementSize = 4
    } = config;

    // Output dimensions (valid convolution, no padding)
    const outputH = inputH - kernelH + 1;
    const outputW = inputW - kernelW + 1;

    // Tensor sizes in elements
    const inputSize = inputH * inputW * channels_in;
    const kernelSize = kernelH * kernelW * channels_in * channels_out;
    const outputSize = outputH * outputW * channels_out;

    // Base addresses for each tensor
    const BASE_INPUT = 0;
    const BASE_KERNEL = inputSize * elementSize;
    const BASE_OUTPUT = (inputSize + kernelSize) * elementSize;

    return {
        name: 'conv2d',
        displayName: '2D Convolution',
        elementSize: elementSize,

        // Dimensions metadata for rendering
        dimensions: {
            inputH, inputW, channels_in,
            kernelH, kernelW,
            outputH, outputW, channels_out
        },

        // Loop dimensions for this operation (6 dimensions, no batch)
        loopDims: ['c_out', 'h_out', 'w_out', 'c_in', 'k_h', 'k_w'],

        // Bounds for each loop dimension
        loopBounds: {
            c_out: channels_out,
            h_out: outputH,
            w_out: outputW,
            c_in: channels_in,
            k_h: kernelH,
            k_w: kernelW
        },

        // All valid loop orderings (subset of permutations - most educational ones)
        loopOrders: {
            // Standard output-major orderings
            'c_out,h_out,w_out,c_in,k_h,k_w': ['c_out', 'h_out', 'w_out', 'c_in', 'k_h', 'k_w'],
            'h_out,w_out,c_out,c_in,k_h,k_w': ['h_out', 'w_out', 'c_out', 'c_in', 'k_h', 'k_w'],
            // Input-reuse orderings (better for input cache)
            'c_in,k_h,k_w,c_out,h_out,w_out': ['c_in', 'k_h', 'k_w', 'c_out', 'h_out', 'w_out'],
            // Kernel-reuse orderings (better for kernel cache)
            'k_h,k_w,c_in,c_out,h_out,w_out': ['k_h', 'k_w', 'c_in', 'c_out', 'h_out', 'w_out']
        },

        // Tensor definitions
        // For convolution, tensors have 3-4 dimensions but we linearize for memory
        tensors: [
            {
                name: 'Input',
                baseAddress: BASE_INPUT,
                // For visualization: show as channels side-by-side, each channel is H×W
                rows: inputH,
                cols: inputW,
                channels: channels_in,
                is3D: true,
                // Layout options for this tensor
                layoutOptions: [
                    { value: 'CHW', label: 'CHW' },
                    { value: 'HWC', label: 'HWC' }
                ],
                // Input access: Input[c_in][h_out + k_h][w_out + k_w]
                getIndices: (iter) => ({
                    channel: iter.c_in,
                    row: iter.h_out + iter.k_h,
                    col: iter.w_out + iter.k_w
                }),
                getTotalElements: () => inputH * inputW * channels_in,
                getLinearIndex: (iter, layout) => {
                    const c = iter.c_in;
                    const h = iter.h_out + iter.k_h;
                    const w = iter.w_out + iter.k_w;
                    if (layout === 'HWC') {
                        // HWC: h * (W * C) + w * C + c
                        return h * (inputW * channels_in) + w * channels_in + c;
                    }
                    // CHW (default): c * (H * W) + h * W + w
                    return c * (inputH * inputW) + h * inputW + w;
                },
                getCoordinatesFromLinear: (linearIdx, layout) => {
                    if (layout === 'HWC') {
                        const h = Math.floor(linearIdx / (inputW * channels_in));
                        const w = Math.floor((linearIdx % (inputW * channels_in)) / channels_in);
                        const c = linearIdx % channels_in;
                        return { channel: c, row: h, col: w };
                    }
                    // CHW
                    const c = Math.floor(linearIdx / (inputH * inputW));
                    const h = Math.floor((linearIdx % (inputH * inputW)) / inputW);
                    const w = linearIdx % inputW;
                    return { channel: c, row: h, col: w };
                }
            },
            {
                name: 'Kernel',
                baseAddress: BASE_KERNEL,
                // For visualization: show as a grid of small kernels (c_out × c_in grid of k_h×k_w)
                rows: kernelH,
                cols: kernelW,
                channels_in: channels_in,
                channels_out: channels_out,
                is4D: true,
                // Layout options for kernel
                layoutOptions: [
                    { value: 'OIHW', label: 'OIHW' },
                    { value: 'HWIO', label: 'HWIO' }
                ],
                // Kernel access: Kernel[c_out][c_in][k_h][k_w]
                getIndices: (iter) => ({
                    c_out: iter.c_out,
                    c_in: iter.c_in,
                    row: iter.k_h,
                    col: iter.k_w
                }),
                getTotalElements: () => kernelH * kernelW * channels_in * channels_out,
                getLinearIndex: (iter, layout) => {
                    const o = iter.c_out;
                    const i = iter.c_in;
                    const h = iter.k_h;
                    const w = iter.k_w;
                    if (layout === 'HWIO') {
                        // HWIO: h * (W * I * O) + w * (I * O) + i * O + o
                        return h * (kernelW * channels_in * channels_out) +
                               w * (channels_in * channels_out) +
                               i * channels_out + o;
                    }
                    // OIHW (default): o * (I * H * W) + i * (H * W) + h * W + w
                    return o * (channels_in * kernelH * kernelW) +
                           i * (kernelH * kernelW) +
                           h * kernelW + w;
                },
                getCoordinatesFromLinear: (linearIdx, layout) => {
                    if (layout === 'HWIO') {
                        const h = Math.floor(linearIdx / (kernelW * channels_in * channels_out));
                        const remainder1 = linearIdx % (kernelW * channels_in * channels_out);
                        const w = Math.floor(remainder1 / (channels_in * channels_out));
                        const remainder2 = remainder1 % (channels_in * channels_out);
                        const i = Math.floor(remainder2 / channels_out);
                        const o = remainder2 % channels_out;
                        return { c_out: o, c_in: i, row: h, col: w };
                    }
                    // OIHW
                    const o = Math.floor(linearIdx / (channels_in * kernelH * kernelW));
                    const remainder1 = linearIdx % (channels_in * kernelH * kernelW);
                    const i = Math.floor(remainder1 / (kernelH * kernelW));
                    const remainder2 = remainder1 % (kernelH * kernelW);
                    const h = Math.floor(remainder2 / kernelW);
                    const w = remainder2 % kernelW;
                    return { c_out: o, c_in: i, row: h, col: w };
                }
            },
            {
                name: 'Output',
                baseAddress: BASE_OUTPUT,
                // For visualization: show as channels side-by-side, each channel is H_out×W_out
                rows: outputH,
                cols: outputW,
                channels: channels_out,
                is3D: true,
                // Layout options for this tensor
                layoutOptions: [
                    { value: 'CHW', label: 'CHW' },
                    { value: 'HWC', label: 'HWC' }
                ],
                // Output access: Output[c_out][h_out][w_out]
                getIndices: (iter) => ({
                    channel: iter.c_out,
                    row: iter.h_out,
                    col: iter.w_out
                }),
                getTotalElements: () => outputH * outputW * channels_out,
                getLinearIndex: (iter, layout) => {
                    const c = iter.c_out;
                    const h = iter.h_out;
                    const w = iter.w_out;
                    if (layout === 'HWC') {
                        // HWC: h * (W * C) + w * C + c
                        return h * (outputW * channels_out) + w * channels_out + c;
                    }
                    // CHW (default): c * (H * W) + h * W + w
                    return c * (outputH * outputW) + h * outputW + w;
                },
                getCoordinatesFromLinear: (linearIdx, layout) => {
                    if (layout === 'HWC') {
                        const h = Math.floor(linearIdx / (outputW * channels_out));
                        const w = Math.floor((linearIdx % (outputW * channels_out)) / channels_out);
                        const c = linearIdx % channels_out;
                        return { channel: c, row: h, col: w };
                    }
                    // CHW
                    const c = Math.floor(linearIdx / (outputH * outputW));
                    const h = Math.floor((linearIdx % (outputH * outputW)) / outputW);
                    const w = linearIdx % outputW;
                    return { channel: c, row: h, col: w };
                }
            }
        ],

        // Inner statement for code display
        codeTemplate: 'Out[c_out][h][w] += In[c_in][h+kh][w+kw] * K[c_out][c_in][kh][kw]',

        // Human-readable operation description
        describeOp: (iter) => {
            const h_in = iter.h_out + iter.k_h;
            const w_in = iter.w_out + iter.k_w;
            return `In[${iter.c_in}][${h_in}][${w_in}] × K[${iter.c_out}][${iter.c_in}][${iter.k_h}][${iter.k_w}] → Out[${iter.c_out}][${iter.h_out}][${iter.w_out}]`;
        },

        // Total iterations
        getTotalIterations: () => channels_out * outputH * outputW * channels_in * kernelH * kernelW,

        // Operators displayed between tensors
        tensorOperators: ['*', '='],

        // Tiling configuration - only spatial dimensions are tiled
        tileableDims: ['h_out', 'w_out'],  // Only tile output spatial dimensions
        tileSizes: [2, 4]                   // 2x2 or 4x4 tiles (output is 6x6)
    };
}

// =============================================================================
// CONSTANTS
// =============================================================================

const MATRIX_SIZE = 12;
const ELEMENT_SIZE = 4; // bytes per element

// =============================================================================
// OPERATION REGISTRY
// =============================================================================

// Registry of available operations
const OPERATIONS = {
    matmul: {
        create: () => createMatmulOperation(MATRIX_SIZE, ELEMENT_SIZE),
        title: 'Matrix Multiplication: Tiling & Cache Visualization',
        defaultLoopOrder: 'ijk'
    },
    conv2d: {
        create: () => createConv2dOperation({ elementSize: ELEMENT_SIZE }),
        title: '2D Convolution: Tiling & Cache Visualization',
        defaultLoopOrder: 'c_out,h_out,w_out,c_in,k_h,k_w'
    }
};

// Current mode and operation
let currentMode = 'matmul';
let operation = OPERATIONS[currentMode].create();

// Note: Use operation.getTotalIterations() dynamically instead of a const
// since the operation can change when switching modes

// Rendering constants
const CELL_SIZE = 20; // pixels per cell
const COLORS = {
    background: '#ffffff',
    grid: '#cccccc',
    tileGrid: '#666666',
    cached: 'rgba(40, 167, 69, 0.6)',
    current: '#000000',
    currentOutline: '#667eea',
    currentSlice: '#667eea'  // border highlight for active slice
};

// =============================================================================
// ISOMETRIC PROJECTION CONFIG
// =============================================================================

const ISO = {
    // Offset per depth layer (how much each slice shifts)
    depthOffsetX: 8,   // pixels to shift right per layer
    depthOffsetY: -6,  // pixels to shift up per layer (negative = up)

    // Transparency for stacked slices
    sliceAlpha: 0.85,           // base alpha for slices
    backSliceAlphaDrop: 0.08,   // additional alpha drop per layer back

    // Minimum gap between slices (on top of offset)
    sliceGap: 2
};

// =============================================================================
// CELL DRAWING HELPERS
// =============================================================================

/**
 * Draw a single cell with optional fill and border.
 * This is the common drawing primitive for all tensor visualizations.
 *
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {number} x - Screen X position
 * @param {number} y - Screen Y position
 * @param {number} width - Cell width
 * @param {number} height - Cell height
 * @param {Object} options - Drawing options
 */
function drawCell(ctx, x, y, width, height, options = {}) {
    const {
        fillColor = null,
        strokeColor = null,
        lineWidth = 1,
        alpha = 1.0
    } = options;

    const prevAlpha = ctx.globalAlpha;
    ctx.globalAlpha = alpha;

    if (fillColor) {
        ctx.fillStyle = fillColor;
        ctx.fillRect(x, y, width, height);
    }

    if (strokeColor) {
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = lineWidth;
        ctx.strokeRect(x, y, width, height);
    }

    ctx.globalAlpha = prevAlpha;
}

/**
 * Draw a filled cell for cached element.
 */
function drawCachedCell(ctx, x, y, alpha = 1.0) {
    drawCell(ctx, x, y, CELL_SIZE, CELL_SIZE, {
        fillColor: COLORS.cached,
        alpha: alpha
    });
}

/**
 * Draw current access indicator.
 */
function drawCurrentAccessCell(ctx, x, y, alpha = 1.0) {
    const prevAlpha = ctx.globalAlpha;
    ctx.globalAlpha = alpha;

    // Inner fill
    ctx.fillStyle = COLORS.current;
    ctx.fillRect(x + 3, y + 3, CELL_SIZE - 6, CELL_SIZE - 6);

    // Outer stroke
    ctx.strokeStyle = COLORS.currentOutline;
    ctx.lineWidth = 2;
    ctx.strokeRect(x + 1, y + 1, CELL_SIZE - 2, CELL_SIZE - 2);

    ctx.globalAlpha = prevAlpha;
}

/**
 * Draw a parallelogram (for isometric slice background/border).
 * Points are defined clockwise from top-left.
 *
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {Array} points - Array of {x, y} points (4 corners)
 * @param {Object} options - Drawing options (fillColor, strokeColor, lineWidth, alpha)
 */
function drawParallelogram(ctx, points, options = {}) {
    const {
        fillColor = null,
        strokeColor = null,
        lineWidth = 1,
        alpha = 1.0
    } = options;

    const prevAlpha = ctx.globalAlpha;
    ctx.globalAlpha = alpha;

    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.closePath();

    if (fillColor) {
        ctx.fillStyle = fillColor;
        ctx.fill();
    }

    if (strokeColor) {
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = lineWidth;
        ctx.stroke();
    }

    ctx.globalAlpha = prevAlpha;
}

/**
 * Calculate screen position for a cell in an isometric slice.
 *
 * @param {number} row - Row in the 2D slice
 * @param {number} col - Column in the 2D slice
 * @param {number} depth - Depth layer (0 = front, higher = back)
 * @param {number} baseX - Base X offset for this tensor
 * @param {number} baseY - Base Y offset for this tensor
 * @returns {{x: number, y: number}} Screen coordinates
 */
function isoPosition(row, col, depth, baseX, baseY) {
    return {
        x: baseX + col * CELL_SIZE + depth * ISO.depthOffsetX,
        y: baseY + row * CELL_SIZE + depth * ISO.depthOffsetY
    };
}

/**
 * Calculate alpha for a slice at given depth.
 * Front slices (depth=0) are most opaque, back slices fade slightly.
 *
 * @param {number} depth - Depth layer (0 = front)
 * @param {number} totalDepth - Total number of layers
 * @returns {number} Alpha value (0-1)
 */
function isoAlpha(depth, totalDepth) {
    // Invert: we draw back-to-front, so depth 0 in drawing order is actually the back
    const distanceFromFront = totalDepth - 1 - depth;
    return Math.max(0.4, ISO.sliceAlpha - distanceFromFront * ISO.backSliceAlphaDrop);
}

// =============================================================================
// APPLICATION STATE
// =============================================================================

/**
 * Initialize tensor state (layouts, stats) from operation definition.
 */
function createTensorState(op) {
    const layouts = {};
    const stats = {};

    for (const tensor of op.tensors) {
        // Use tensor-specific default layout if available, otherwise 'row'
        if (tensor.layoutOptions && tensor.layoutOptions.length > 0) {
            layouts[tensor.name] = tensor.layoutOptions[0].value;
        } else {
            layouts[tensor.name] = 'row';
        }
        stats[tensor.name] = { accesses: 0, hits: 0 };
    }

    return { layouts, stats };
}

const tensorState = createTensorState(operation);

const state = {
    // Configuration (from UI)
    loopOrder: 'ijk',
    tilingEnabled: false,
    tileSize: 4,
    layouts: tensorState.layouts,  // { A: 'row', B: 'row', C: 'row' }
    elementsPerLine: 16,  // elements per cache line
    numCacheLines: 4,     // number of cache lines

    // Simulation state
    currentIteration: 0,
    isPlaying: false,
    speed: 10,

    // Generated data
    iterations: [],
    cache: null,

    // Statistics (per-tensor)
    stats: tensorState.stats,  // { A: {accesses, hits}, B: {...}, C: {...} }

    // History for timeline
    history: []  // Array of { A: hit/miss, B: hit/miss, C: hit/miss }
};

// Canvas contexts
let canvasContexts = {};  // { A: ctx, B: ctx, C: ctx }
let ctxTimeline;
let ctxMemoryLayout;

// Animation state
let animationId = null;
let lastFrameTime = 0;
let snapshots = [];

// =============================================================================
// ITERATION GENERATOR
// =============================================================================

/**
 * Generate iteration sequence for non-tiled execution.
 *
 * @param {Object} op - Operation definition
 * @param {string} loopOrder - Loop nesting order (e.g., 'ijk')
 * @returns {Array} Sequence of index objects
 */
function generateIterations(op, loopOrder) {
    const iterations = [];
    const order = op.loopOrders[loopOrder];

    if (!order) {
        throw new Error(`Unknown loop order: ${loopOrder}`);
    }

    const bounds = op.loopBounds;

    // Recursive helper to generate N-dimensional nested loops
    function nestLoops(depth, currentIndices) {
        if (depth === order.length) {
            // Leaf: create iteration object with all loop dimensions
            const iter = {};
            for (const dim of op.loopDims) {
                iter[dim] = currentIndices[dim];
            }
            iterations.push(iter);
            return;
        }

        const dim = order[depth];
        const bound = bounds[dim];
        for (let i = 0; i < bound; i++) {
            currentIndices[dim] = i;
            nestLoops(depth + 1, currentIndices);
        }
    }

    nestLoops(0, {});
    return iterations;
}

/**
 * Generate iteration sequence for tiled execution.
 * Supports partial tiling where only some dimensions are tiled.
 *
 * Loop structure for partial tiling:
 * 1. Non-tiled dims before first tiled dim (outer loops)
 * 2. Tile loops for all tiled dims
 * 3. Non-tiled dims after first tiled dim (between tile and element loops)
 * 4. Element loops for all tiled dims (innermost)
 *
 * @param {Object} op - Operation definition
 * @param {string} loopOrder - Loop nesting order
 * @param {number} tileSize - Tile dimension
 * @returns {Array} Sequence of index objects with tile info
 */
function generateTiledIterations(op, loopOrder, tileSize) {
    const order = op.loopOrders[loopOrder];

    if (!order) {
        throw new Error(`Unknown loop order: ${loopOrder}`);
    }

    const bounds = op.loopBounds;
    const tileableDims = new Set(op.tileableDims || op.loopDims);  // Default: all dims tileable

    // Find first tiled dimension in the loop order
    const firstTiledIdx = order.findIndex(d => tileableDims.has(d));
    if (firstTiledIdx === -1) {
        // No tileable dimensions - fall back to regular iteration
        return generateIterations(op, loopOrder);
    }

    // Build loop specification
    // Each entry: { dim, type: 'simple'|'tile'|'element', ... }
    const loopSpec = [];

    // Phase 1: Non-tiled dims before first tiled dim (stay as outer loops)
    for (let i = 0; i < firstTiledIdx; i++) {
        loopSpec.push({ dim: order[i], type: 'simple', bound: bounds[order[i]] });
    }

    // Phase 2: Tile loops for all tiled dims (in their order of appearance)
    for (const dim of order) {
        if (tileableDims.has(dim)) {
            loopSpec.push({ dim: 't' + dim, type: 'tile', tiledDim: dim, bound: bounds[dim], step: tileSize });
        }
    }

    // Phase 3: Everything from first tiled dim onwards, maintaining relative order
    // Tiled dims become element loops, non-tiled dims stay as simple loops
    // This preserves the original loop semantics (e.g., for conv2d: h_out, w_out before c_in, k_h, k_w)
    for (let i = firstTiledIdx; i < order.length; i++) {
        const dim = order[i];
        if (tileableDims.has(dim)) {
            loopSpec.push({ dim: dim, type: 'element', tiledDim: dim, bound: bounds[dim], tileSize: tileSize });
        } else {
            loopSpec.push({ dim: dim, type: 'simple', bound: bounds[dim] });
        }
    }

    // Generate iterations using the loop spec
    const iterations = [];

    function nest(depth, currentIndices) {
        if (depth === loopSpec.length) {
            // Leaf: create iteration object
            const iter = {};
            for (const dim of op.loopDims) {
                iter[dim] = currentIndices[dim];
                if (tileableDims.has(dim)) {
                    iter['t' + dim] = currentIndices['t' + dim];
                    iter['l' + dim] = currentIndices[dim] - currentIndices['t' + dim];
                }
            }
            iterations.push(iter);
            return;
        }

        const spec = loopSpec[depth];

        if (spec.type === 'simple') {
            for (let v = 0; v < spec.bound; v++) {
                currentIndices[spec.dim] = v;
                nest(depth + 1, currentIndices);
            }
        } else if (spec.type === 'tile') {
            for (let t = 0; t < spec.bound; t += spec.step) {
                currentIndices[spec.dim] = t;
                nest(depth + 1, currentIndices);
            }
        } else if (spec.type === 'element') {
            const tileBase = currentIndices['t' + spec.tiledDim];
            const tileEnd = Math.min(tileBase + spec.tileSize, spec.bound);
            for (let e = tileBase; e < tileEnd; e++) {
                currentIndices[spec.dim] = e;
                nest(depth + 1, currentIndices);
            }
        }
    }

    nest(0, {});
    return iterations;
}

/**
 * Generate all iterations based on current configuration.
 */
function generateAllIterations() {
    if (state.tilingEnabled) {
        return generateTiledIterations(operation, state.loopOrder, state.tileSize);
    }
    return generateIterations(operation, state.loopOrder);
}

// =============================================================================
// CACHE MODEL
// =============================================================================

/**
 * LRU Cache Simulator
 */
class CacheSimulator {
    constructor(capacityBytes, lineSize) {
        this.capacityBytes = capacityBytes;
        this.lineSize = lineSize;
        this.maxLines = Math.floor(capacityBytes / lineSize);
        this.lines = [];
        this.totalAccesses = 0;
        this.hits = 0;
        this.misses = 0;
    }

    getLineAddress(address) {
        return Math.floor(address / this.lineSize) * this.lineSize;
    }

    access(address) {
        this.totalAccesses++;
        const lineAddr = this.getLineAddress(address);
        const index = this.lines.indexOf(lineAddr);

        if (index !== -1) {
            this.lines.splice(index, 1);
            this.lines.push(lineAddr);
            this.hits++;
            return true;
        } else {
            this.misses++;
            if (this.lines.length >= this.maxLines) {
                this.lines.shift();
            }
            this.lines.push(lineAddr);
            return false;
        }
    }

    isAddressCached(address) {
        const lineAddr = this.getLineAddress(address);
        return this.lines.includes(lineAddr);
    }

    reset() {
        this.lines = [];
        this.totalAccesses = 0;
        this.hits = 0;
        this.misses = 0;
    }

    snapshot() {
        return {
            lines: [...this.lines],
            totalAccesses: this.totalAccesses,
            hits: this.hits,
            misses: this.misses
        };
    }

    restore(snap) {
        this.lines = [...snap.lines];
        this.totalAccesses = snap.totalAccesses;
        this.hits = snap.hits;
        this.misses = snap.misses;
    }
}

// =============================================================================
// MEMORY ADDRESS CALCULATION
// =============================================================================

/**
 * Calculate memory address for a tensor element.
 *
 * @param {Object} tensor - Tensor definition from operation
 * @param {number} row - Row index
 * @param {number} col - Column index
 * @param {string} layout - 'row' or 'col'
 * @returns {number} Memory address in bytes
 */
function getTensorAddress(tensor, row, col, layout) {
    // For simple 2D tensors (matmul)
    const size = tensor.rows || operation.size;
    if (layout === 'col') {
        return tensor.baseAddress + (col * size + row) * operation.elementSize;
    }
    return tensor.baseAddress + (row * size + col) * operation.elementSize;
}

/**
 * Get memory address for a tensor access given current iteration.
 * Supports both 2D tensors (matmul) and multi-dimensional tensors (convolution).
 */
function getAccessAddress(tensor, iter) {
    // If tensor has custom linear index function, use it (for multi-dimensional tensors)
    if (tensor.getLinearIndex) {
        const layout = state.layouts[tensor.name];
        const linearIndex = tensor.getLinearIndex(iter, layout);
        return tensor.baseAddress + linearIndex * operation.elementSize;
    }

    // Fall back to 2D row/col calculation
    const indices = tensor.getIndices(iter);
    const layout = state.layouts[tensor.name];
    return getTensorAddress(tensor, indices.row, indices.col, layout);
}

// =============================================================================
// UI GENERATION
// =============================================================================

/**
 * Generate all dynamic UI elements from the current operation definition.
 * Called on init and when switching modes.
 */
function generateTensorUI() {
    generateLayoutControls();
    generateTensorCanvases();
    generateTensorStatsCards();
    updateTimelineLabel();
}

/**
 * Generate layout control dropdowns for each tensor.
 */
function generateLayoutControls() {
    const container = document.getElementById('layoutControls');
    container.innerHTML = '';

    for (const tensor of operation.tensors) {
        const item = document.createElement('div');
        item.className = 'layout-item';

        // Use tensor-specific layout options if available
        let optionsHTML;
        if (tensor.layoutOptions && tensor.layoutOptions.length > 0) {
            optionsHTML = tensor.layoutOptions
                .map(opt => `<option value="${opt.value}">${opt.label}</option>`)
                .join('');
        } else {
            // Default row/col options for matmul tensors
            optionsHTML = `
                <option value="row">Row-major</option>
                <option value="col">Col-major</option>
            `;
        }

        item.innerHTML = `
            <span>${tensor.name}</span>
            <select id="layout${tensor.name}">${optionsHTML}</select>
        `;
        container.appendChild(item);
    }
}

/**
 * Generate tensor canvas wrappers and canvases.
 */
// Gap between channels in multi-channel tensor visualization
const CHANNEL_GAP = 4;

/**
 * Calculate canvas dimensions for a tensor based on its shape.
 */
function getTensorCanvasSize(tensor) {
    const rows = tensor.rows;
    const cols = tensor.cols;

    if (tensor.is4D) {
        // 4D tensor (kernel): show as rows of isometric 3D stacks
        // Each row is one c_out value, showing c_in channels stacked
        const c_out = tensor.channels_out;
        const c_in = tensor.channels_in;

        // Width: base grid + isometric depth offset for all channels
        const baseWidth = cols * CELL_SIZE;
        const depthWidth = (c_in - 1) * ISO.depthOffsetX;
        const width = baseWidth + depthWidth + 20;  // padding

        // Height: rows of 3D stacks with gaps + isometric vertical offset
        const baseHeight = rows * CELL_SIZE;
        const depthHeight = Math.abs((c_in - 1) * ISO.depthOffsetY);
        const rowHeight = baseHeight + depthHeight + CHANNEL_GAP;
        const height = c_out * rowHeight + 20;  // padding

        return { width, height };
    } else if (tensor.is3D) {
        // 3D tensor: isometric stacked slices
        const channels = tensor.channels;

        // Width: base grid + isometric depth offset for all channels
        const baseWidth = cols * CELL_SIZE;
        const depthWidth = (channels - 1) * ISO.depthOffsetX;
        const width = baseWidth + depthWidth + 20;  // padding for labels

        // Height: base grid + isometric vertical offset (goes upward)
        const baseHeight = rows * CELL_SIZE;
        const depthHeight = Math.abs((channels - 1) * ISO.depthOffsetY);
        const height = baseHeight + depthHeight + 20;  // padding for labels

        return { width, height };
    } else {
        // 2D tensor (matmul): square
        const size = rows * CELL_SIZE;
        return { width: size, height: size };
    }
}

function generateTensorCanvases() {
    const container = document.getElementById('tensorsContainer');
    container.innerHTML = '';

    const tensors = operation.tensors;
    const operators = operation.tensorOperators || [];

    for (let i = 0; i < tensors.length; i++) {
        const tensor = tensors[i];
        const { width, height } = getTensorCanvasSize(tensor);

        // Add tensor wrapper
        const wrapper = document.createElement('div');
        wrapper.className = 'matrix-wrapper';
        wrapper.innerHTML = `
            <div class="matrix-header">
                <span class="matrix-name">${tensor.name}</span>
                <span class="matrix-stats" id="stats${tensor.name}">mem:0 hit:0</span>
            </div>
            <canvas id="matrix${tensor.name}" class="matrix-canvas" width="${width}" height="${height}"></canvas>
        `;
        container.appendChild(wrapper);

        // Add operator between tensors (if not last tensor)
        if (i < operators.length) {
            const operatorSpan = document.createElement('span');
            operatorSpan.className = 'operator';
            operatorSpan.textContent = operators[i];
            container.appendChild(operatorSpan);
        }
    }
}

/**
 * Generate per-tensor statistics cards.
 */
function generateTensorStatsCards() {
    const container = document.getElementById('tensorStatsContainer');
    container.innerHTML = '';

    for (const tensor of operation.tensors) {
        const card = document.createElement('div');
        card.className = 'matrix-stat-card';
        card.innerHTML = `
            <div class="name">${tensor.name}</div>
            <div class="value" id="detailStats${tensor.name}">0/0 hits</div>
        `;
        container.appendChild(card);
    }
}

/**
 * Update the timeline label with current tensor names.
 */
function updateTimelineLabel() {
    const names = operation.tensors.map(t => t.name).join(' | ');
    document.getElementById('timelineLabel').textContent = `Cache Hit Global Timeline (${names})`;
}

// =============================================================================
// RENDERING
// =============================================================================

/**
 * Initialize canvas contexts for all tensors.
 */
function initCanvases() {
    for (const tensor of operation.tensors) {
        const canvas = document.getElementById('matrix' + tensor.name);
        if (canvas) {
            canvasContexts[tensor.name] = canvas.getContext('2d');
        }
    }

    ctxTimeline = document.getElementById('timeline').getContext('2d');
    const timelineCanvas = document.getElementById('timeline');
    timelineCanvas.width = timelineCanvas.offsetWidth;
    timelineCanvas.height = 60;

    ctxMemoryLayout = document.getElementById('memoryLayout').getContext('2d');
    const memoryLayoutCanvas = document.getElementById('memoryLayout');
    memoryLayoutCanvas.width = memoryLayoutCanvas.offsetWidth;
    memoryLayoutCanvas.height = 80;
}

/**
 * Render a single tensor grid.
 * Supports 2D, 3D (with channels), and 4D (kernel) tensors.
 *
 * @param {Object} tensor - Tensor definition from operation
 * @param {Object|null} currentIndices - Current access indices or null
 */
function renderTensor(tensor, currentIndices) {
    const ctx = canvasContexts[tensor.name];
    if (!ctx) return;

    const canvas = ctx.canvas;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Background
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (tensor.is4D) {
        render4DTensor(ctx, tensor, currentIndices);
    } else if (tensor.is3D) {
        render3DTensor(ctx, tensor, currentIndices);
    } else {
        render2DTensor(ctx, tensor, currentIndices);
    }
}

/**
 * Render a 2D tensor (matmul style).
 */
function render2DTensor(ctx, tensor, currentIndices) {
    const rows = tensor.rows;
    const cols = tensor.cols;

    // Cached elements
    if (state.cache) {
        for (let row = 0; row < rows; row++) {
            for (let col = 0; col < cols; col++) {
                if (isElementInCache2D(tensor, row, col)) {
                    ctx.fillStyle = COLORS.cached;
                    ctx.fillRect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE);
                }
            }
        }
    }

    // Grid lines
    drawGrid(ctx, rows, cols, 0, 0);

    // Tile boundaries
    if (state.tilingEnabled && state.tileSize > 1) {
        drawTileBoundaries(ctx, rows, cols, 0, 0);
    }

    // Current access indicator
    if (currentIndices && currentIndices.row !== undefined) {
        drawCurrentAccess(ctx, currentIndices.row, currentIndices.col, 0, 0);
    }
}

/**
 * Render a 3D tensor (channels side-by-side).
 */
function render3DTensor(ctx, tensor, currentIndices) {
    const rows = tensor.rows;
    const cols = tensor.cols;
    const channels = tensor.channels;

    // Base position accounting for isometric offset (front slice at bottom-left)
    const baseX = 5;
    const baseY = Math.abs((channels - 1) * ISO.depthOffsetY) + 5;

    // Draw back-to-front for proper layering
    for (let c = channels - 1; c >= 0; c--) {
        const depth = channels - 1 - c;  // depth 0 = back slice
        const alpha = isoAlpha(depth, channels);

        // Calculate slice offset from base position
        const sliceOffsetX = c * ISO.depthOffsetX;
        const sliceOffsetY = c * ISO.depthOffsetY;

        // Draw slice background (subtle fill to show slice boundary)
        const sliceCorners = [
            { x: baseX + sliceOffsetX, y: baseY + sliceOffsetY },
            { x: baseX + sliceOffsetX + cols * CELL_SIZE, y: baseY + sliceOffsetY },
            { x: baseX + sliceOffsetX + cols * CELL_SIZE, y: baseY + sliceOffsetY + rows * CELL_SIZE },
            { x: baseX + sliceOffsetX, y: baseY + sliceOffsetY + rows * CELL_SIZE }
        ];
        drawParallelogram(ctx, sliceCorners, {
            fillColor: '#f8f8f8',
            strokeColor: COLORS.grid,
            lineWidth: 1,
            alpha: alpha
        });

        // Cached elements for this channel
        if (state.cache) {
            for (let row = 0; row < rows; row++) {
                for (let col = 0; col < cols; col++) {
                    if (isElementInCache3D(tensor, c, row, col)) {
                        const pos = isoPosition(row, col, c, baseX, baseY);
                        drawCachedCell(ctx, pos.x, pos.y, alpha);
                    }
                }
            }
        }

        // Grid lines for this channel
        drawIsoGrid(ctx, rows, cols, baseX + sliceOffsetX, baseY + sliceOffsetY, alpha);

        // Channel label (on front edge of slice)
        ctx.globalAlpha = alpha;
        ctx.fillStyle = '#666';
        ctx.font = '9px sans-serif';
        ctx.fillText(`c${c}`, baseX + sliceOffsetX + 2, baseY + sliceOffsetY + rows * CELL_SIZE + 10);
        ctx.globalAlpha = 1.0;
    }

    // Tile boundaries on front slice (c=0) when tiling is enabled
    // Only show on Input tensor, not Output (Output tiles don't align nicely with dimensions)
    if (state.tilingEnabled && state.tileSize > 1 && tensor.name !== 'Output') {
        drawIsoTileBoundaries(ctx, rows, cols, baseX, baseY, state.tileSize);
    }

    // Current slice highlight and access indicator (always on top with full opacity)
    if (currentIndices && currentIndices.channel !== undefined) {
        const c = currentIndices.channel;

        // Highlight current slice border
        const sliceOffsetX = c * ISO.depthOffsetX;
        const sliceOffsetY = c * ISO.depthOffsetY;
        const sliceCorners = [
            { x: baseX + sliceOffsetX, y: baseY + sliceOffsetY },
            { x: baseX + sliceOffsetX + cols * CELL_SIZE, y: baseY + sliceOffsetY },
            { x: baseX + sliceOffsetX + cols * CELL_SIZE, y: baseY + sliceOffsetY + rows * CELL_SIZE },
            { x: baseX + sliceOffsetX, y: baseY + sliceOffsetY + rows * CELL_SIZE }
        ];
        drawParallelogram(ctx, sliceCorners, {
            strokeColor: COLORS.currentSlice,
            lineWidth: 2,
            alpha: 1.0
        });

        // Current cell indicator
        const pos = isoPosition(currentIndices.row, currentIndices.col, c, baseX, baseY);
        drawCurrentAccessCell(ctx, pos.x, pos.y, 1.0);
    }
}

/**
 * Draw grid lines for an isometric slice.
 */
function drawIsoGrid(ctx, rows, cols, xOffset, yOffset, alpha) {
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;

    // Vertical lines
    for (let i = 0; i <= cols; i++) {
        ctx.beginPath();
        ctx.moveTo(xOffset + i * CELL_SIZE, yOffset);
        ctx.lineTo(xOffset + i * CELL_SIZE, yOffset + rows * CELL_SIZE);
        ctx.stroke();
    }
    // Horizontal lines
    for (let i = 0; i <= rows; i++) {
        ctx.beginPath();
        ctx.moveTo(xOffset, yOffset + i * CELL_SIZE);
        ctx.lineTo(xOffset + cols * CELL_SIZE, yOffset + i * CELL_SIZE);
        ctx.stroke();
    }
    ctx.globalAlpha = 1.0;
}

/**
 * Draw tile boundaries for an isometric slice.
 */
function drawIsoTileBoundaries(ctx, rows, cols, xOffset, yOffset, tileSize) {
    ctx.strokeStyle = COLORS.tileGrid;
    ctx.lineWidth = 2;

    // Vertical tile lines
    for (let i = 0; i <= cols; i += tileSize) {
        ctx.beginPath();
        ctx.moveTo(xOffset + i * CELL_SIZE, yOffset);
        ctx.lineTo(xOffset + i * CELL_SIZE, yOffset + rows * CELL_SIZE);
        ctx.stroke();
    }
    // Horizontal tile lines
    for (let i = 0; i <= rows; i += tileSize) {
        ctx.beginPath();
        ctx.moveTo(xOffset, yOffset + i * CELL_SIZE);
        ctx.lineTo(xOffset + cols * CELL_SIZE, yOffset + i * CELL_SIZE);
        ctx.stroke();
    }
}

/**
 * Render a 4D tensor (kernel) as rows of isometric 3D stacks.
 * Each row represents one c_out value, with c_in slices stacked isometrically.
 */
function render4DTensor(ctx, tensor, currentIndices) {
    const kRows = tensor.rows;  // kernel height
    const kCols = tensor.cols;  // kernel width
    const c_in = tensor.channels_in;
    const c_out = tensor.channels_out;

    // Calculate row height including isometric depth offset
    const baseHeight = kRows * CELL_SIZE;
    const depthHeight = Math.abs((c_in - 1) * ISO.depthOffsetY);
    const rowHeight = baseHeight + depthHeight + CHANNEL_GAP + 5;

    // Base position
    const baseX = 25;  // room for co labels

    for (let co = 0; co < c_out; co++) {
        // Y position for this output channel row
        const rowBaseY = 10 + co * rowHeight + depthHeight;

        // c_out label for this row
        ctx.fillStyle = '#666';
        ctx.font = '9px sans-serif';
        ctx.fillText(`co${co}`, 2, rowBaseY + kRows * CELL_SIZE / 2 + 3);

        // Draw c_in slices back-to-front for this output channel
        for (let ci = c_in - 1; ci >= 0; ci--) {
            const depth = c_in - 1 - ci;  // depth 0 = back slice
            const alpha = isoAlpha(depth, c_in);

            // Slice offset from isometric projection
            const sliceOffsetX = ci * ISO.depthOffsetX;
            const sliceOffsetY = ci * ISO.depthOffsetY;

            const sliceX = baseX + sliceOffsetX;
            const sliceY = rowBaseY + sliceOffsetY;

            // Draw slice background
            const sliceCorners = [
                { x: sliceX, y: sliceY },
                { x: sliceX + kCols * CELL_SIZE, y: sliceY },
                { x: sliceX + kCols * CELL_SIZE, y: sliceY + kRows * CELL_SIZE },
                { x: sliceX, y: sliceY + kRows * CELL_SIZE }
            ];
            drawParallelogram(ctx, sliceCorners, {
                fillColor: '#f8f8f8',
                strokeColor: COLORS.grid,
                lineWidth: 1,
                alpha: alpha
            });

            // Cached elements for this kernel slice
            if (state.cache) {
                for (let row = 0; row < kRows; row++) {
                    for (let col = 0; col < kCols; col++) {
                        if (isElementInCache4D(tensor, co, ci, row, col)) {
                            const cellX = sliceX + col * CELL_SIZE;
                            const cellY = sliceY + row * CELL_SIZE;
                            drawCachedCell(ctx, cellX, cellY, alpha);
                        }
                    }
                }
            }

            // Grid lines for this kernel slice
            drawIsoGrid(ctx, kRows, kCols, sliceX, sliceY, alpha);
        }
    }

    // Current slice highlight and access indicator (always on top with full opacity)
    if (currentIndices && currentIndices.c_out !== undefined) {
        const co = currentIndices.c_out;
        const ci = currentIndices.c_in;
        const rowBaseY = 10 + co * rowHeight + depthHeight;

        const sliceX = baseX + ci * ISO.depthOffsetX;
        const sliceY = rowBaseY + ci * ISO.depthOffsetY;

        // Highlight current slice border
        const sliceCorners = [
            { x: sliceX, y: sliceY },
            { x: sliceX + kCols * CELL_SIZE, y: sliceY },
            { x: sliceX + kCols * CELL_SIZE, y: sliceY + kRows * CELL_SIZE },
            { x: sliceX, y: sliceY + kRows * CELL_SIZE }
        ];
        drawParallelogram(ctx, sliceCorners, {
            strokeColor: COLORS.currentSlice,
            lineWidth: 2,
            alpha: 1.0
        });

        // Current cell indicator
        const cellX = sliceX + currentIndices.col * CELL_SIZE;
        const cellY = sliceY + currentIndices.row * CELL_SIZE;
        drawCurrentAccessCell(ctx, cellX, cellY, 1.0);
    }
}

/**
 * Draw grid lines for a tensor slice.
 */
function drawGrid(ctx, rows, cols, xOffset, yOffset) {
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;

    for (let i = 0; i <= cols; i++) {
        ctx.beginPath();
        ctx.moveTo(xOffset + i * CELL_SIZE, yOffset);
        ctx.lineTo(xOffset + i * CELL_SIZE, yOffset + rows * CELL_SIZE);
        ctx.stroke();
    }
    for (let i = 0; i <= rows; i++) {
        ctx.beginPath();
        ctx.moveTo(xOffset, yOffset + i * CELL_SIZE);
        ctx.lineTo(xOffset + cols * CELL_SIZE, yOffset + i * CELL_SIZE);
        ctx.stroke();
    }
}

/**
 * Draw tile boundaries for a tensor slice.
 */
function drawTileBoundaries(ctx, rows, cols, xOffset, yOffset) {
    ctx.strokeStyle = COLORS.tileGrid;
    ctx.lineWidth = 2;

    for (let i = 0; i <= cols; i += state.tileSize) {
        ctx.beginPath();
        ctx.moveTo(xOffset + i * CELL_SIZE, yOffset);
        ctx.lineTo(xOffset + i * CELL_SIZE, yOffset + rows * CELL_SIZE);
        ctx.stroke();
    }
    for (let i = 0; i <= rows; i += state.tileSize) {
        ctx.beginPath();
        ctx.moveTo(xOffset, yOffset + i * CELL_SIZE);
        ctx.lineTo(xOffset + cols * CELL_SIZE, yOffset + i * CELL_SIZE);
        ctx.stroke();
    }
}

/**
 * Draw current access indicator at a cell position.
 */
function drawCurrentAccess(ctx, row, col, xOffset, yOffset) {
    const x = xOffset + col * CELL_SIZE;
    const y = yOffset + row * CELL_SIZE;

    ctx.fillStyle = COLORS.current;
    ctx.fillRect(x + 3, y + 3, CELL_SIZE - 6, CELL_SIZE - 6);
    ctx.strokeStyle = COLORS.currentOutline;
    ctx.lineWidth = 2;
    ctx.strokeRect(x + 1, y + 1, CELL_SIZE - 2, CELL_SIZE - 2);
}

/**
 * Check if element is in cache for 2D tensor.
 */
function isElementInCache2D(tensor, row, col) {
    const layout = state.layouts[tensor.name];
    const address = getTensorAddress(tensor, row, col, layout);
    return state.cache.isAddressCached(address);
}

/**
 * Check if element is in cache for 3D tensor.
 */
function isElementInCache3D(tensor, channel, row, col) {
    const layout = state.layouts[tensor.name];
    let linearIndex;

    if (layout === 'HWC') {
        // HWC: row * (cols * channels) + col * channels + channel
        linearIndex = row * (tensor.cols * tensor.channels) + col * tensor.channels + channel;
    } else {
        // CHW (default): channel * (rows * cols) + row * cols + col
        linearIndex = channel * (tensor.rows * tensor.cols) + row * tensor.cols + col;
    }

    const address = tensor.baseAddress + linearIndex * operation.elementSize;
    return state.cache.isAddressCached(address);
}

/**
 * Check if element is in cache for 4D tensor (kernel).
 */
function isElementInCache4D(tensor, c_out, c_in, row, col) {
    const layout = state.layouts[tensor.name];
    let linearIndex;

    if (layout === 'HWIO') {
        // HWIO: row * (cols * c_in * c_out) + col * (c_in * c_out) + c_in * c_out + c_out
        linearIndex = row * (tensor.cols * tensor.channels_in * tensor.channels_out) +
                      col * (tensor.channels_in * tensor.channels_out) +
                      c_in * tensor.channels_out + c_out;
    } else {
        // OIHW (default): c_out * (c_in * rows * cols) + c_in * (rows * cols) + row * cols + col
        linearIndex = c_out * (tensor.channels_in * tensor.rows * tensor.cols) +
                      c_in * (tensor.rows * tensor.cols) +
                      row * tensor.cols + col;
    }

    const address = tensor.baseAddress + linearIndex * operation.elementSize;
    return state.cache.isAddressCached(address);
}

/**
 * Render all tensors.
 */
function renderAllTensors() {
    const iter = state.iterations[state.currentIteration];

    for (const tensor of operation.tensors) {
        const indices = iter ? tensor.getIndices(iter) : null;
        renderTensor(tensor, indices);
    }
}

/**
 * Render the cache hit timeline.
 */
function renderTimeline() {
    const canvas = ctxTimeline.canvas;
    const width = canvas.width;
    const height = canvas.height;
    const numTensors = operation.tensors.length;

    ctxTimeline.fillStyle = '#1a1a2e';
    ctxTimeline.fillRect(0, 0, width, height);

    if (state.history.length === 0) return;

    const rowHeight = height / numTensors;
    const labelOffset = 15;
    const barWidthScaled = (width - labelOffset) / operation.getTotalIterations();

    // Labels
    ctxTimeline.fillStyle = '#666';
    ctxTimeline.font = '10px monospace';
    operation.tensors.forEach((tensor, idx) => {
        ctxTimeline.fillText(tensor.name, 2, idx * rowHeight + rowHeight / 2 + 3);
    });

    // History bars
    for (let i = 0; i < state.history.length; i++) {
        const x = labelOffset + i * barWidthScaled;
        const h = state.history[i];
        const barW = Math.max(1, barWidthScaled - 0.5);

        operation.tensors.forEach((tensor, idx) => {
            ctxTimeline.fillStyle = h[tensor.name] ? '#28a745' : '#dc3545';
            ctxTimeline.fillRect(x, idx * rowHeight + 2, barW, rowHeight - 4);
        });
    }

    // Current position indicator
    if (state.currentIteration > 0) {
        const x = labelOffset + (state.currentIteration - 1) * barWidthScaled;
        ctxTimeline.strokeStyle = '#667eea';
        ctxTimeline.lineWidth = 2;
        ctxTimeline.beginPath();
        ctxTimeline.moveTo(x + barWidthScaled, 0);
        ctxTimeline.lineTo(x + barWidthScaled, height);
        ctxTimeline.stroke();
    }
}

/**
 * Get linear index for a tensor element based on layout.
 * Row-major: row * size + col
 * Col-major: col * size + row
 */
function getLinearIndex(row, col, layout) {
    const size = operation.size;
    if (layout === 'col') {
        return col * size + row;
    }
    return row * size + col;
}

/**
 * Check if an element is in cache given tensor and coordinates.
 * Works for 2D, 3D, and 4D tensors.
 */
function isElementInCacheByCoords(tensor, coords) {
    if (tensor.is4D) {
        return isElementInCache4D(tensor, coords.c_out, coords.c_in, coords.row, coords.col);
    } else if (tensor.is3D) {
        return isElementInCache3D(tensor, coords.channel, coords.row, coords.col);
    } else {
        return isElementInCache2D(tensor, coords.row, coords.col);
    }
}

/**
 * Render the memory layout visualization.
 * Shows linear address space for each tensor with cache residency.
 * Supports both 2D tensors (matmul) and 3D/4D tensors (conv2d).
 */
function renderMemoryLayout() {
    const canvas = ctxMemoryLayout.canvas;
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctxMemoryLayout.fillStyle = '#1a1a2e';
    ctxMemoryLayout.fillRect(0, 0, width, height);

    const numTensors = operation.tensors.length;

    // Calculate total elements across all tensors for proportional sizing
    const tensorSizes = operation.tensors.map(t => t.getTotalElements());
    const maxElements = Math.max(...tensorSizes);

    const rowHeight = height / numTensors;
    const labelOffset = 45;  // Space for tensor names
    const barWidth = width - labelOffset;

    // Get current iteration for highlighting
    const iter = state.iterations[state.currentIteration];

    // Draw each tensor's memory layout
    operation.tensors.forEach((tensor, tensorIdx) => {
        const y = tensorIdx * rowHeight;
        const layout = state.layouts[tensor.name];
        const numElements = tensor.getTotalElements();
        // All tensors use the same element width for visual consistency
        const elemWidth = barWidth / maxElements;

        const xStart = labelOffset;

        // Label
        ctxMemoryLayout.fillStyle = '#888';
        ctxMemoryLayout.font = '9px monospace';
        ctxMemoryLayout.fillText(tensor.name, 2, y + rowHeight / 2 + 3);

        // Current access position (linear index)
        let currentLinearIndex = -1;
        if (iter) {
            currentLinearIndex = tensor.getLinearIndex(iter, layout);
        }

        // Draw each element
        for (let linearIdx = 0; linearIdx < numElements; linearIdx++) {
            const x = xStart + linearIdx * elemWidth;

            // Convert linear index back to coordinates
            const coords = tensor.getCoordinatesFromLinear(linearIdx, layout);

            // Check if this element is in cache
            const inCache = isElementInCacheByCoords(tensor, coords);

            // Draw element
            if (linearIdx === currentLinearIndex) {
                // Current access - black with outline
                ctxMemoryLayout.fillStyle = '#000000';
                ctxMemoryLayout.fillRect(x, y + 3, Math.max(1, elemWidth - 0.5), rowHeight - 6);
                ctxMemoryLayout.strokeStyle = '#667eea';
                ctxMemoryLayout.lineWidth = 1;
                ctxMemoryLayout.strokeRect(x, y + 2, Math.max(1, elemWidth), rowHeight - 4);
            } else if (inCache) {
                // In cache - green
                ctxMemoryLayout.fillStyle = '#28a745';
                ctxMemoryLayout.fillRect(x, y + 3, Math.max(1, elemWidth - 0.5), rowHeight - 6);
            } else {
                // Not in cache - dark background
                ctxMemoryLayout.fillStyle = '#2a2a4a';
                ctxMemoryLayout.fillRect(x, y + 3, Math.max(1, elemWidth - 0.5), rowHeight - 6);
            }
        }

        // Draw cache line boundaries
        const elementsPerLine = state.elementsPerLine;
        ctxMemoryLayout.strokeStyle = '#555';
        ctxMemoryLayout.lineWidth = 1;
        for (let lineStart = 0; lineStart <= numElements; lineStart += elementsPerLine) {
            const x = xStart + lineStart * elemWidth;
            ctxMemoryLayout.beginPath();
            ctxMemoryLayout.moveTo(x, y + 1);
            ctxMemoryLayout.lineTo(x, y + rowHeight - 1);
            ctxMemoryLayout.stroke();
        }

        // Draw tensor boundary (vertical line at end of tensor memory)
        const xEnd = xStart + numElements * elemWidth;
        ctxMemoryLayout.strokeStyle = '#888';
        ctxMemoryLayout.lineWidth = 2;
        ctxMemoryLayout.beginPath();
        ctxMemoryLayout.moveTo(xEnd, y + 1);
        ctxMemoryLayout.lineTo(xEnd, y + rowHeight - 1);
        ctxMemoryLayout.stroke();
    });
}

/**
 * Update statistics display.
 */
function updateStatsDisplay() {
    // Per-tensor stats in header
    for (const tensor of operation.tensors) {
        const statsEl = document.getElementById('stats' + tensor.name);
        if (statsEl) {
            const s = state.stats[tensor.name];
            statsEl.textContent = `mem:${s.accesses} hit:${s.hits}`;
        }
    }

    // Overall stats
    let totalAccesses = 0;
    let totalHits = 0;
    for (const tensor of operation.tensors) {
        totalAccesses += state.stats[tensor.name].accesses;
        totalHits += state.stats[tensor.name].hits;
    }
    const hitRate = totalAccesses > 0 ? (totalHits / totalAccesses * 100).toFixed(0) : 0;

    document.getElementById('totalMem').textContent = totalAccesses;
    document.getElementById('totalHits').textContent = totalHits;
    document.getElementById('hitRate').textContent = hitRate + '%';

    // Detailed per-tensor stats
    for (const tensor of operation.tensors) {
        const detailEl = document.getElementById('detailStats' + tensor.name);
        if (detailEl) {
            const s = state.stats[tensor.name];
            detailEl.textContent = `${s.hits}/${s.accesses} hits`;
        }
    }
}

/**
 * Update current state display.
 */
function updateStateDisplay() {
    const iter = state.iterations[state.currentIteration];

    document.getElementById('currentIteration').textContent = `${state.currentIteration} of ${operation.getTotalIterations()}`;

    if (iter) {
        // Build indices string
        let indicesStr = '';
        const tileableDims = new Set(operation.tileableDims || operation.loopDims);
        const hasTileInfo = state.tilingEnabled && operation.tileableDims &&
                           iter['t' + operation.tileableDims[0]] !== undefined;

        if (hasTileInfo) {
            // Show tile indices for tiled dims, then all element indices
            const tileParts = operation.loopDims
                .filter(d => tileableDims.has(d))
                .map(d => `t${d}=${iter['t' + d]}`);
            const elemParts = operation.loopDims.map(d => `${d}=${iter[d]}`);
            indicesStr = tileParts.join(', ') + ', ' + elemParts.join(', ');
        } else {
            indicesStr = operation.loopDims.map(d => `${d}=${iter[d]}`).join(', ');
        }
        document.getElementById('currentIndices').textContent = indicesStr;
        document.getElementById('currentOp').textContent = operation.describeOp(iter);
    } else {
        document.getElementById('currentIndices').textContent = '-';
        document.getElementById('currentOp').textContent = '-';
    }

    document.getElementById('jumpIteration').max = operation.getTotalIterations() - 1;
    document.getElementById('totalIterations').textContent = operation.getTotalIterations();
}

/**
 * Full render of all visualization components.
 */
function render() {
    renderAllTensors();
    renderMemoryLayout();
    renderTimeline();
    updateStatsDisplay();
    updateStateDisplay();
}

// =============================================================================
// SIMULATION LOGIC
// =============================================================================

/**
 * Execute one simulation step.
 * Uses the operation abstraction to determine memory accesses.
 */
function executeStep() {
    if (state.currentIteration >= state.iterations.length) {
        return null;
    }

    const iter = state.iterations[state.currentIteration];
    const result = {};

    // Access each tensor according to the operation's access pattern
    for (const tensor of operation.tensors) {
        const address = getAccessAddress(tensor, iter);
        const hit = state.cache.access(address);

        state.stats[tensor.name].accesses++;
        state.stats[tensor.name].hits += hit ? 1 : 0;
        result[tensor.name] = hit;
    }

    state.history.push(result);
    state.currentIteration++;

    return result;
}

/**
 * Reset simulation to initial state.
 */
function resetSimulation() {
    stopAnimation();

    state.currentIteration = 0;
    state.history = [];
    snapshots = [];

    // Reset stats for all tensors
    for (const tensor of operation.tensors) {
        state.stats[tensor.name] = { accesses: 0, hits: 0 };
    }

    if (state.cache) {
        state.cache.reset();
    }

    document.getElementById('playPauseBtn').textContent = '▶';
    render();
    updateCodeDisplay();
}

/**
 * Jump to a specific iteration.
 */
function jumpToIteration(targetIteration) {
    targetIteration = Math.max(0, Math.min(targetIteration, state.iterations.length));

    if (targetIteration < state.currentIteration) {
        resetSimulation();
    }

    while (state.currentIteration < targetIteration) {
        executeStep();
    }

    render();
    updateCodeDisplay();
}

/**
 * Step forward one iteration.
 */
function stepForward() {
    if (state.currentIteration < state.iterations.length) {
        snapshots.push({
            cache: state.cache.snapshot(),
            stats: JSON.parse(JSON.stringify(state.stats)),
            historyLength: state.history.length,
            iteration: state.currentIteration
        });

        executeStep();
        render();
        updateCodeDisplay();
    }
}

/**
 * Step backward one iteration.
 */
function stepBackward() {
    if (snapshots.length > 0) {
        const snapshot = snapshots.pop();
        state.cache.restore(snapshot.cache);
        state.stats = snapshot.stats;
        state.history = state.history.slice(0, snapshot.historyLength);
        state.currentIteration = snapshot.iteration;

        render();
        updateCodeDisplay();
    }
}

/**
 * Animation loop.
 */
function animate(timestamp) {
    if (!state.isPlaying) return;

    const elapsed = timestamp - lastFrameTime;
    const interval = 1000 / state.speed;

    if (elapsed >= interval) {
        if (state.currentIteration < state.iterations.length) {
            if (snapshots.length > 100) {
                snapshots = snapshots.slice(-50);
            }

            snapshots.push({
                cache: state.cache.snapshot(),
                stats: JSON.parse(JSON.stringify(state.stats)),
                historyLength: state.history.length,
                iteration: state.currentIteration
            });

            executeStep();
            render();
            updateCodeDisplay();
            lastFrameTime = timestamp;
        } else {
            stopAnimation();
        }
    }

    if (state.isPlaying) {
        animationId = requestAnimationFrame(animate);
    }
}

function startAnimation() {
    if (!state.isPlaying && state.currentIteration < state.iterations.length) {
        state.isPlaying = true;
        document.getElementById('playPauseBtn').textContent = '⏸';
        lastFrameTime = performance.now();
        animationId = requestAnimationFrame(animate);
    }
}

function stopAnimation() {
    state.isPlaying = false;
    document.getElementById('playPauseBtn').textContent = '▶';
    if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }
}

function togglePlayPause() {
    if (state.isPlaying) {
        stopAnimation();
    } else {
        startAnimation();
    }
}

/**
 * Apply configuration from UI controls.
 *
 * Note on cache line count: With 3-tensor operations (A, B, C), fewer than 4 cache
 * lines causes severe thrashing. Each iteration accesses all 3 tensors, so with
 * N < 4 lines, at least one tensor's line is evicted before its next access.
 * This makes spatial locality benefits impossible to observe. UI restricts to >= 4 lines.
 */

// =============================================================================
// MODE SWITCHING
// =============================================================================

/**
 * Switch to a different operation mode (matmul, conv2d, etc.)
 * @param {string} mode - The mode to switch to
 */
function switchMode(mode) {
    if (mode === currentMode) return;

    const opConfig = OPERATIONS[mode];
    if (!opConfig || !opConfig.create) {
        console.log(`Mode "${mode}" not yet implemented`);
        return;
    }

    // Stop any running animation
    stopAnimation();

    currentMode = mode;

    // Create new operation
    operation = opConfig.create();

    // Reinitialize tensor state for new operation
    const newTensorState = createTensorState(operation);
    state.layouts = newTensorState.layouts;
    state.stats = newTensorState.stats;

    // Update UI
    document.querySelectorAll('.mode-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.mode === mode);
    });
    document.getElementById('pageTitle').textContent = opConfig.title;

    // Regenerate dynamic UI elements
    generateTensorUI();
    generateLoopOrderOptions();
    generateTileSizeOptions();

    // Reinitialize canvases
    initCanvases();

    // Set default loop order for this operation
    state.loopOrder = opConfig.defaultLoopOrder;
    document.getElementById('loopOrder').value = state.loopOrder;

    // Apply configuration (resets simulation)
    applyConfiguration();

    console.log(`Switched to mode: ${mode}`);
    console.log(`Tensors: ${operation.tensors.map(t => t.name).join(', ')}`);
    console.log(`Total iterations: ${operation.getTotalIterations()}`);
}

/**
 * Generate loop order dropdown options based on current operation.
 */
function generateLoopOrderOptions() {
    const select = document.getElementById('loopOrder');
    select.innerHTML = '';

    for (const [key, order] of Object.entries(operation.loopOrders)) {
        const option = document.createElement('option');
        option.value = key;
        option.textContent = order.join(' → ');
        select.appendChild(option);
    }
}

/**
 * Generate tile size dropdown options based on current operation.
 */
function generateTileSizeOptions() {
    const select = document.getElementById('tileSize');
    const currentValue = parseInt(select.value);
    select.innerHTML = '';

    const tileSizes = operation.tileSizes || [2, 4, 6];
    const tileableDims = operation.tileableDims || operation.loopDims;

    for (const size of tileSizes) {
        const option = document.createElement('option');
        option.value = size;
        // Show which dimensions are tiled
        if (tileableDims.length < operation.loopDims.length) {
            // Partial tiling - show tiled dimensions
            option.textContent = `${size}×${size} (${tileableDims.join(', ')})`;
        } else {
            option.textContent = `${size}×${size}`;
        }
        select.appendChild(option);
    }

    // Try to preserve current selection, otherwise use first available
    if (tileSizes.includes(currentValue)) {
        select.value = currentValue;
    } else {
        select.value = tileSizes[0];
    }
}

// =============================================================================
// CONFIGURATION
// =============================================================================

function applyConfiguration() {
    state.loopOrder = document.getElementById('loopOrder').value;
    state.tilingEnabled = document.getElementById('tilingEnabled').checked;
    state.tileSize = parseInt(document.getElementById('tileSize').value);

    // Cache configuration (minimum 4 lines to avoid thrashing with 3 tensors)
    state.elementsPerLine = parseInt(document.getElementById('elementsPerLine').value);
    state.numCacheLines = parseInt(document.getElementById('numCacheLines').value);

    // Read layouts for all tensors
    for (const tensor of operation.tensors) {
        const layoutEl = document.getElementById('layout' + tensor.name);
        if (layoutEl) {
            state.layouts[tensor.name] = layoutEl.value;
        }
    }

    // Compute cache parameters
    const cacheLineSize = state.elementsPerLine * ELEMENT_SIZE;
    const cacheCapacity = cacheLineSize * state.numCacheLines;

    state.iterations = generateAllIterations();
    state.cache = new CacheSimulator(cacheCapacity, cacheLineSize);

    resetSimulation();
}

/**
 * Update the byte equivalent displays for cache configuration.
 */
function updateCacheDisplays() {
    const elementsPerLine = parseInt(document.getElementById('elementsPerLine').value);
    const numCacheLines = parseInt(document.getElementById('numCacheLines').value);

    const lineBytes = elementsPerLine * ELEMENT_SIZE;
    const totalBytes = lineBytes * numCacheLines;

    document.getElementById('lineSizeBytes').textContent = `= ${lineBytes}B`;
    document.getElementById('cacheSizeBytes').textContent = `= ${totalBytes}B`;
}

// =============================================================================
// CODE DISPLAY
// =============================================================================

/**
 * Get the loop order array for the current configuration.
 */
function getLoopOrderArray() {
    return operation.loopOrders[state.loopOrder] || operation.loopDims;
}

/**
 * Update the loop code display.
 */
function updateCodeDisplay() {
    const codeDiv = document.getElementById('codeDisplay');
    const order = getLoopOrderArray();

    if (state.tilingEnabled) {
        codeDiv.innerHTML = generateTiledCodeHTML(order, state.tileSize);
    } else {
        codeDiv.innerHTML = generateNonTiledCodeHTML(order);
    }
}

/**
 * Generate HTML for non-tiled loop code.
 */
function generateNonTiledCodeHTML(order) {
    const iter = state.iterations[state.currentIteration];
    const bounds = operation.loopBounds;
    let html = '';
    const numLoops = order.length;

    // Generate indentation dynamically for any number of loops
    const getIndent = (level) => '  '.repeat(level);

    for (let level = 0; level < numLoops; level++) {
        const varName = order[level];
        const bound = bounds[varName];
        const isInnermost = level === numLoops - 1;
        const isCurrent = isInnermost && state.currentIteration < state.iterations.length;

        html += `<div class="code-line${isCurrent ? ' current' : ''}">`;
        html += `${getIndent(level)}<span class="code-keyword">for</span> `;
        html += `<span class="code-var">${varName}</span> `;
        html += `<span class="code-keyword">in</span> `;
        html += `<span class="code-number">0</span>..<span class="code-number">${bound}</span>:`;
        if (isCurrent && iter) {
            html += ` <span class="code-comment">← ${varName}=${iter[varName]}</span>`;
        }
        html += '</div>';
    }

    html += `<div class="code-line">${getIndent(numLoops)}${operation.codeTemplate}</div>`;
    return html;
}

/**
 * Generate HTML for tiled loop code.
 * Supports partial tiling where only some dimensions are tiled.
 *
 * Loop structure matches generateTiledIterations:
 * 1. Non-tiled dims before first tiled dim (outer)
 * 2. Tile loops for all tiled dims
 * 3. Everything from first tiled dim onwards in original order:
 *    - Tiled dims as element loops
 *    - Non-tiled dims as simple loops
 */
function generateTiledCodeHTML(order, tileSize) {
    const iter = state.iterations[state.currentIteration];
    const bounds = operation.loopBounds;
    const tileableDims = new Set(operation.tileableDims || operation.loopDims);
    let html = '';
    let indentLevel = 0;

    const getIndent = (level) => '  '.repeat(level);

    // Find first tiled dimension
    const firstTiledIdx = order.findIndex(d => tileableDims.has(d));
    if (firstTiledIdx === -1) {
        // No tileable dims - show as non-tiled
        return generateNonTiledCodeHTML(order);
    }

    // Phase 1: Non-tiled dims before first tiled dim
    for (let i = 0; i < firstTiledIdx; i++) {
        const dim = order[i];
        const bound = bounds[dim];
        html += `<div class="code-line">`;
        html += `${getIndent(indentLevel)}<span class="code-keyword">for</span> `;
        html += `<span class="code-var">${dim}</span> `;
        html += `<span class="code-keyword">in</span> `;
        html += `<span class="code-number">0</span>..<span class="code-number">${bound}</span>:`;
        if (iter) {
            html += ` <span class="code-comment">← ${dim}=${iter[dim]}</span>`;
        }
        html += '</div>';
        indentLevel++;
    }

    // Phase 2: Tile loops for tiled dims (in their order of appearance)
    for (const dim of order) {
        if (tileableDims.has(dim)) {
            const varName = 't' + dim;
            const bound = bounds[dim];
            html += `<div class="code-line">`;
            html += `${getIndent(indentLevel)}<span class="code-keyword">for</span> `;
            html += `<span class="code-var">${varName}</span> `;
            html += `<span class="code-keyword">in</span> `;
            html += `<span class="code-number">0</span>..<span class="code-number">${bound}</span> `;
            html += `<span class="code-keyword">step</span> <span class="code-number">${tileSize}</span>:`;
            if (iter && iter[varName] !== undefined) {
                html += ` <span class="code-comment">← ${varName}=${iter[varName]}</span>`;
            }
            html += '</div>';
            indentLevel++;
        }
    }

    // Phase 3: Everything from first tiled dim onwards, maintaining relative order
    // Tiled dims become element loops, non-tiled dims stay as simple loops
    for (let i = firstTiledIdx; i < order.length; i++) {
        const dim = order[i];
        const bound = bounds[dim];
        const isLastLoop = i === order.length - 1;
        const isCurrent = isLastLoop && state.currentIteration < state.iterations.length;

        if (tileableDims.has(dim)) {
            // Element loop for tiled dimension
            const tileVar = 't' + dim;
            html += `<div class="code-line${isCurrent ? ' current' : ''}">`;
            html += `${getIndent(indentLevel)}<span class="code-keyword">for</span> `;
            html += `<span class="code-var">${dim}</span> `;
            html += `<span class="code-keyword">in</span> `;
            html += `<span class="code-var">${tileVar}</span>..<span class="code-var">${tileVar}</span>+<span class="code-number">${tileSize}</span>:`;
            if (isCurrent && iter) {
                html += ` <span class="code-comment">← ${dim}=${iter[dim]}</span>`;
            }
            html += '</div>';
        } else {
            // Simple loop for non-tiled dimension
            html += `<div class="code-line${isCurrent ? ' current' : ''}">`;
            html += `${getIndent(indentLevel)}<span class="code-keyword">for</span> `;
            html += `<span class="code-var">${dim}</span> `;
            html += `<span class="code-keyword">in</span> `;
            html += `<span class="code-number">0</span>..<span class="code-number">${bound}</span>:`;
            if (isCurrent && iter) {
                html += ` <span class="code-comment">← ${dim}=${iter[dim]}</span>`;
            }
            html += '</div>';
        }
        indentLevel++;
    }

    html += `<div class="code-line">${getIndent(indentLevel)}${operation.codeTemplate}</div>`;
    return html;
}

// =============================================================================
// UI EVENT HANDLERS
// =============================================================================

function setupEventHandlers() {
    // Mode switching tabs
    document.querySelectorAll('.mode-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            switchMode(tab.dataset.mode);
        });
    });

    // Tiling toggle
    document.getElementById('tilingEnabled').addEventListener('change', (e) => {
        document.getElementById('tileSize').disabled = !e.target.checked;
    });

    // Cache configuration - update byte displays on change
    document.getElementById('elementsPerLine').addEventListener('change', updateCacheDisplays);
    document.getElementById('numCacheLines').addEventListener('change', updateCacheDisplays);

    // Configuration
    const applyBtn = document.getElementById('applyConfig');
    applyBtn.addEventListener('click', () => {
        applyConfiguration();
        applyBtn.classList.remove('needs-apply');
    });

    // Highlight apply button when config changes
    // Static config inputs (always present)
    const staticConfigInputs = [
        'loopOrder', 'tilingEnabled', 'tileSize',
        'elementsPerLine', 'numCacheLines'
    ];
    staticConfigInputs.forEach(id => {
        document.getElementById(id).addEventListener('change', () => {
            applyBtn.classList.add('needs-apply');
        });
    });

    // Dynamic layout controls - use event delegation on container
    document.getElementById('layoutControls').addEventListener('change', (e) => {
        if (e.target.tagName === 'SELECT') {
            applyBtn.classList.add('needs-apply');
        }
    });

    // Playback
    document.getElementById('playPauseBtn').addEventListener('click', togglePlayPause);
    document.getElementById('stepFwdBtn').addEventListener('click', stepForward);
    document.getElementById('stepBackBtn').addEventListener('click', stepBackward);
    document.getElementById('resetBtn').addEventListener('click', resetSimulation);

    // Speed slider
    document.getElementById('speedSlider').addEventListener('input', (e) => {
        const speed = Math.round(Math.pow(10, e.target.value / 50));
        state.speed = speed;
        document.getElementById('speedDisplay').textContent = speed + 'x';
    });

    // Jump to iteration
    document.getElementById('jumpIteration').addEventListener('change', (e) => {
        const target = Math.min(Math.max(0, parseInt(e.target.value) || 0), operation.getTotalIterations() - 1);
        jumpToIteration(target);
        e.target.value = state.currentIteration;
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

        switch (e.code) {
            case 'Space':
                e.preventDefault();
                togglePlayPause();
                break;
            case 'ArrowRight':
                e.preventDefault();
                stepForward();
                break;
            case 'ArrowLeft':
                e.preventDefault();
                stepBackward();
                break;
            case 'KeyR':
                e.preventDefault();
                resetSimulation();
                break;
        }
    });

    // Window resize
    window.addEventListener('resize', () => {
        const timelineCanvas = document.getElementById('timeline');
        timelineCanvas.width = timelineCanvas.offsetWidth;
        const memoryLayoutCanvas = document.getElementById('memoryLayout');
        memoryLayoutCanvas.width = memoryLayoutCanvas.offsetWidth;
        renderMemoryLayout();
        renderTimeline();
    });
}

// =============================================================================
// GUIDED TOUR
// =============================================================================

/**
 * Tour step definitions.
 * Each step targets a CSS selector and provides educational content.
 * position: 'top' | 'bottom' | 'left' | 'right' | 'auto'
 */
/**
 * Get tour steps based on current operation.
 * Steps are mostly generic but the first one describes the operation.
 */
function getTourSteps() {
    const opIntro = operation.name === 'matmul'
        ? 'We compute C = A × B, stepping through all i,j,k iterations.'
        : 'We compute Output = Input * Kernel, stepping through all output positions, input channels, and kernel positions.';

    const tensorDesc = operation.name === 'matmul'
        ? 'Each cell shows a matrix element.'
        : 'For convolution, tensors have channels shown side-by-side. The kernel shows all c_out × c_in filter slices.';

    return [
        {
            target: '.matrices-container',
            title: operation.displayName,
            content: `${opIntro} ${tensorDesc} Green highlighting indicates the element is currently in cache. The black dot shows which element is being accessed.`,
            position: 'bottom'
        },
        {
            target: '#memoryLayout',
            title: 'Linear Memory Layout',
            content: 'Tensors are stored as flat arrays in memory. This bar shows each tensor\'s linear address space. Green = in cache. Vertical lines mark cache line boundaries. Watch how access patterns create different locality behaviors.',
            position: 'top'
        },
        {
            target: '.config-panel',
            title: 'Configuration',
            content: 'Control the simulation parameters here. Loop order changes which index varies fastest. Data layout (row/col major) affects how indices map to linear memory addresses. Cache settings control the simulated cache size.',
            position: 'bottom'
        },
        {
            target: '.playback-controls',
            title: 'Playback Controls',
            content: 'Step through iterations one by one, or play continuously. Use the speed slider to control animation speed. You can also jump to any iteration directly.',
            position: 'top'
        },
        {
            target: '.stats-bar',
            title: 'Cache Statistics',
            content: 'Track total memory accesses and cache hits. The hit rate shows cache efficiency. Better locality = higher hit rate = faster real-world performance.',
            position: 'top'
        },
        {
            target: '#timeline',
            title: 'Cache Hit Timeline',
            content: 'History of cache hits (green) and misses (red) for each tensor over time. Patterns here reveal locality behavior: clustered green = good locality, scattered red = poor locality.',
            position: 'top'
        },
        {
            target: '.side-panel',
            title: 'Loop Structure & State',
            content: 'See the actual loop code being executed, with current indices highlighted. The state panel shows exactly which iteration and operation is happening.',
            position: 'left'
        }
    ];
}

/**
 * Tour state and controller.
 */
const tour = {
    active: false,
    currentStep: 0,
    highlightedElement: null,

    start() {
        this.active = true;
        this.currentStep = 0;
        document.getElementById('tourOverlay').classList.remove('hidden');
        document.getElementById('tourTooltip').classList.remove('hidden');
        this.showStep(0);
    },

    end() {
        this.active = false;
        document.getElementById('tourOverlay').classList.add('hidden');
        document.getElementById('tourTooltip').classList.add('hidden');
        this.clearHighlight();
    },

    showStep(index) {
        if (index < 0 || index >= getTourSteps().length) return;

        this.currentStep = index;
        const step = getTourSteps()[index];

        // Update content
        document.getElementById('tourTitle').textContent = step.title;
        document.getElementById('tourContent').textContent = step.content;
        document.getElementById('tourStepIndicator').textContent = `${index + 1} / ${getTourSteps().length}`;

        // Update navigation buttons
        document.getElementById('tourPrev').disabled = index === 0;
        const nextBtn = document.getElementById('tourNext');
        nextBtn.textContent = index === getTourSteps().length - 1 ? 'Finish' : 'Next';

        // Highlight target element
        this.highlightElement(step.target);

        // Position tooltip
        this.positionTooltip(step.target, step.position);
    },

    highlightElement(selector) {
        this.clearHighlight();
        const element = document.querySelector(selector);
        if (element) {
            element.classList.add('tour-highlight');
            this.highlightedElement = element;
            // Scroll element into view if needed
            element.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    },

    clearHighlight() {
        if (this.highlightedElement) {
            this.highlightedElement.classList.remove('tour-highlight');
            this.highlightedElement = null;
        }
    },

    positionTooltip(selector, preferredPosition) {
        const tooltip = document.getElementById('tourTooltip');
        const target = document.querySelector(selector);
        if (!target) return;

        const targetRect = target.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();
        const padding = 16;
        const arrowSpace = 12;

        // Calculate available space in each direction
        const spaceTop = targetRect.top;
        const spaceBottom = window.innerHeight - targetRect.bottom;
        const spaceLeft = targetRect.left;
        const spaceRight = window.innerWidth - targetRect.right;

        // Determine best position
        let position = preferredPosition;
        if (position === 'auto') {
            // Choose position with most space
            const spaces = { top: spaceTop, bottom: spaceBottom, left: spaceLeft, right: spaceRight };
            position = Object.entries(spaces).sort((a, b) => b[1] - a[1])[0][0];
        }

        // Check if preferred position has enough space, otherwise flip
        const tooltipHeight = tooltipRect.height;
        const tooltipWidth = tooltipRect.width;

        if (position === 'bottom' && spaceBottom < tooltipHeight + padding) {
            position = 'top';
        } else if (position === 'top' && spaceTop < tooltipHeight + padding) {
            position = 'bottom';
        } else if (position === 'left' && spaceLeft < tooltipWidth + padding) {
            position = 'right';
        } else if (position === 'right' && spaceRight < tooltipWidth + padding) {
            position = 'left';
        }

        // Calculate final position
        let top, left;

        switch (position) {
            case 'top':
                top = targetRect.top - tooltipHeight - arrowSpace;
                left = targetRect.left + (targetRect.width - tooltipWidth) / 2;
                break;
            case 'bottom':
                top = targetRect.bottom + arrowSpace;
                left = targetRect.left + (targetRect.width - tooltipWidth) / 2;
                break;
            case 'left':
                top = targetRect.top + (targetRect.height - tooltipHeight) / 2;
                left = targetRect.left - tooltipWidth - arrowSpace;
                break;
            case 'right':
                top = targetRect.top + (targetRect.height - tooltipHeight) / 2;
                left = targetRect.right + arrowSpace;
                break;
        }

        // Clamp to viewport
        top = Math.max(padding, Math.min(top, window.innerHeight - tooltipHeight - padding));
        left = Math.max(padding, Math.min(left, window.innerWidth - tooltipWidth - padding));

        tooltip.style.top = `${top}px`;
        tooltip.style.left = `${left}px`;
    },

    next() {
        if (this.currentStep < getTourSteps().length - 1) {
            this.showStep(this.currentStep + 1);
        } else {
            this.end();
        }
    },

    prev() {
        if (this.currentStep > 0) {
            this.showStep(this.currentStep - 1);
        }
    }
};

/**
 * Set up tour event handlers.
 */
function setupTourHandlers() {
    document.getElementById('helpBtn').addEventListener('click', () => tour.start());
    document.getElementById('tourClose').addEventListener('click', () => tour.end());
    document.getElementById('tourNext').addEventListener('click', () => tour.next());
    document.getElementById('tourPrev').addEventListener('click', () => tour.prev());

    // Close tour on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.code === 'Escape' && tour.active) {
            tour.end();
        }
    });

    // Reposition tooltip on window resize
    window.addEventListener('resize', () => {
        if (tour.active) {
            const step = getTourSteps()[tour.currentStep];
            tour.positionTooltip(step.target, step.position);
        }
    });
}

// =============================================================================
// INITIALIZATION
// =============================================================================

function init() {
    // Skip initialization if essential DOM elements are missing (e.g., in test environment)
    if (!document.getElementById('tensorsContainer')) {
        console.log('Skipping UI initialization (test mode)');
        return;
    }

    console.log('Cache Visualizer initialized');
    console.log(`Operation: ${operation.displayName}`);
    console.log(`Tensors: ${operation.tensors.map(t => t.name).join(', ')}`);

    // Generate dynamic UI elements from operation definition
    generateTensorUI();
    generateLoopOrderOptions();
    generateTileSizeOptions();

    initCanvases();
    setupEventHandlers();
    setupTourHandlers();
    applyConfiguration();

    // Draw attention to help button for new users (3 pulses over 3 seconds)
    const helpBtn = document.getElementById('helpBtn');
    helpBtn.classList.add('attention');
    helpBtn.addEventListener('animationend', () => {
        helpBtn.classList.remove('attention');
    });
}

// Browser: initialize on DOM ready
if (typeof document !== 'undefined') {
    document.addEventListener('DOMContentLoaded', init);
}

// Node.js: export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        createMatmulOperation,
        createConv2dOperation,
        generateIterations,
        generateTiledIterations,
        CacheSimulator,
        getLinearIndex,
        getTensorAddress
    };
}
