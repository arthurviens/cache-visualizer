/**
 * Iteration generation for tensor operations
 * Supports both non-tiled and tiled execution patterns
 */

/**
 * Generate iteration sequence for non-tiled execution.
 *
 * @param {Object} op - Operation definition
 * @param {string} loopOrder - Loop nesting order (e.g., 'ijk')
 * @returns {Array} Sequence of index objects
 */
export function generateIterations(op, loopOrder) {
    const iterations = [];
    const order = op.loopOrders[loopOrder];

    if (!order) {
        throw new Error(`Unknown loop order: ${loopOrder}`);
    }

    const bounds = op.loopBounds;

    function nestLoops(depth, currentIndices) {
        if (depth === order.length) {
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
export function generateTiledIterations(op, loopOrder, tileSize) {
    const order = op.loopOrders[loopOrder];

    if (!order) {
        throw new Error(`Unknown loop order: ${loopOrder}`);
    }

    const bounds = op.loopBounds;
    const tileableDims = new Set(op.tileableDims || op.loopDims);

    const firstTiledIdx = order.findIndex(d => tileableDims.has(d));
    if (firstTiledIdx === -1) {
        return generateIterations(op, loopOrder);
    }

    const loopSpec = [];

    // Phase 1: Non-tiled dims before first tiled dim
    for (let i = 0; i < firstTiledIdx; i++) {
        loopSpec.push({ dim: order[i], type: 'simple', bound: bounds[order[i]] });
    }

    // Phase 2: Tile loops for all tiled dims
    for (const dim of order) {
        if (tileableDims.has(dim)) {
            loopSpec.push({ dim: 't' + dim, type: 'tile', tiledDim: dim, bound: bounds[dim], step: tileSize });
        }
    }

    // Phase 3: Everything from first tiled dim onwards
    for (let i = firstTiledIdx; i < order.length; i++) {
        const dim = order[i];
        if (tileableDims.has(dim)) {
            loopSpec.push({ dim: dim, type: 'element', tiledDim: dim, bound: bounds[dim], tileSize: tileSize });
        } else {
            loopSpec.push({ dim: dim, type: 'simple', bound: bounds[dim] });
        }
    }

    const iterations = [];

    function nest(depth, currentIndices) {
        if (depth === loopSpec.length) {
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
