/**
 * Memory address calculation utilities
 */

import { MATRIX_SIZE } from '../rendering/config.js';

/**
 * Calculate linear index for a 2D tensor element.
 * Used for row-major/column-major layout calculations.
 *
 * @param {number} row - Row index
 * @param {number} col - Column index
 * @param {string} layout - 'row' or 'col'
 * @param {number} size - Matrix dimension (default MATRIX_SIZE)
 * @returns {number} Linear index
 */
export function getLinearIndex(row, col, layout, size = MATRIX_SIZE) {
    if (layout === 'col') {
        return col * size + row;
    }
    return row * size + col;
}

/**
 * Calculate memory address for a tensor element.
 *
 * @param {Object} tensor - Tensor definition from operation
 * @param {number} row - Row index
 * @param {number} col - Column index
 * @param {string} layout - 'row' or 'col'
 * @param {number} elementSize - Bytes per element
 * @returns {number} Memory address in bytes
 */
export function getTensorAddress(tensor, row, col, layout, elementSize = 4) {
    const size = tensor.rows || MATRIX_SIZE;
    if (layout === 'col') {
        return tensor.baseAddress + (col * size + row) * elementSize;
    }
    return tensor.baseAddress + (row * size + col) * elementSize;
}

/**
 * Get memory address for a tensor access given current iteration.
 * Supports both 2D tensors (matmul) and multi-dimensional tensors (convolution).
 *
 * @param {Object} tensor - Tensor definition
 * @param {Object} iter - Current iteration indices
 * @param {Object} layouts - Layout map { tensorName: layout }
 * @param {number} elementSize - Bytes per element
 * @returns {number} Memory address in bytes
 */
export function getAccessAddress(tensor, iter, layouts, elementSize) {
    if (tensor.getLinearIndex) {
        const layout = layouts[tensor.name];
        const linearIndex = tensor.getLinearIndex(iter, layout);
        return tensor.baseAddress + linearIndex * elementSize;
    }

    const indices = tensor.getIndices(iter);
    const layout = layouts[tensor.name];
    return getTensorAddress(tensor, indices.row, indices.col, layout, elementSize);
}
