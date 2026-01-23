/**
 * Tensor rendering functions
 * Handles 2D, 3D, and 4D tensor visualization
 */

import { CELL_SIZE, COLORS, ISO, CHANNEL_GAP } from './config.js';
import {
    drawCachedCell, drawCurrentAccessCell, drawParallelogram,
    isoPosition, isoAlpha, drawGrid, drawTileBoundaries,
    drawIsoGrid, drawIsoTileBoundaries, drawCurrentAccess
} from './primitives.js';

/**
 * Calculate canvas dimensions for a tensor based on its shape.
 */
export function getTensorCanvasSize(tensor) {
    const rows = tensor.rows;
    const cols = tensor.cols;

    if (tensor.is4D) {
        const c_out = tensor.channels_out;
        const c_in = tensor.channels_in;

        const baseWidth = cols * CELL_SIZE;
        const depthWidth = (c_in - 1) * ISO.depthOffsetX;
        const width = baseWidth + depthWidth + 20;

        const baseHeight = rows * CELL_SIZE;
        const depthHeight = Math.abs((c_in - 1) * ISO.depthOffsetY);
        const rowHeight = baseHeight + depthHeight + CHANNEL_GAP;
        const height = c_out * rowHeight + 20;

        return { width, height };
    } else if (tensor.is3D) {
        const channels = tensor.channels;

        const baseWidth = cols * CELL_SIZE;
        const depthWidth = (channels - 1) * ISO.depthOffsetX;
        const width = baseWidth + depthWidth + 20;

        const baseHeight = rows * CELL_SIZE;
        const depthHeight = Math.abs((channels - 1) * ISO.depthOffsetY);
        const height = baseHeight + depthHeight + 20;

        return { width, height };
    } else {
        const size = rows * CELL_SIZE;
        return { width: size, height: size };
    }
}

/**
 * Check if element is in cache for 2D tensor.
 * @returns {number | null} Cache level if cached, null if not
 */
export function isElementInCache2D(tensor, row, col, cache, layouts, elementSize) {
    const layout = layouts[tensor.name];
    const size = tensor.rows;
    let linearIndex;
    if (layout === 'col') {
        linearIndex = col * size + row;
    } else {
        linearIndex = row * size + col;
    }
    const address = tensor.baseAddress + linearIndex * elementSize;
    return cache.isAddressCached(address);
}

/**
 * Check if element is in cache for 3D tensor.
 * @returns {number | null} Cache level if cached, null if not
 */
export function isElementInCache3D(tensor, channel, row, col, cache, layouts, elementSize) {
    const layout = layouts[tensor.name];
    let linearIndex;

    if (layout === 'HWC') {
        linearIndex = row * (tensor.cols * tensor.channels) + col * tensor.channels + channel;
    } else {
        linearIndex = channel * (tensor.rows * tensor.cols) + row * tensor.cols + col;
    }

    const address = tensor.baseAddress + linearIndex * elementSize;
    return cache.isAddressCached(address);
}

/**
 * Check if element is in cache for 4D tensor (kernel).
 * @returns {number | null} Cache level if cached, null if not
 */
export function isElementInCache4D(tensor, c_out, c_in, row, col, cache, layouts, elementSize) {
    const layout = layouts[tensor.name];
    let linearIndex;

    if (layout === 'HWIO') {
        linearIndex = row * (tensor.cols * tensor.channels_in * tensor.channels_out) +
                      col * (tensor.channels_in * tensor.channels_out) +
                      c_in * tensor.channels_out + c_out;
    } else {
        linearIndex = c_out * (tensor.channels_in * tensor.rows * tensor.cols) +
                      c_in * (tensor.rows * tensor.cols) +
                      row * tensor.cols + col;
    }

    const address = tensor.baseAddress + linearIndex * elementSize;
    return cache.isAddressCached(address);
}

/**
 * Check if an element is in cache given tensor and coordinates.
 * @returns {number | null} Cache level if cached, null if not
 */
export function isElementInCacheByCoords(tensor, coords, cache, layouts, elementSize) {
    if (tensor.is4D) {
        return isElementInCache4D(tensor, coords.c_out, coords.c_in, coords.row, coords.col, cache, layouts, elementSize);
    } else if (tensor.is3D) {
        return isElementInCache3D(tensor, coords.channel, coords.row, coords.col, cache, layouts, elementSize);
    } else {
        return isElementInCache2D(tensor, coords.row, coords.col, cache, layouts, elementSize);
    }
}

/**
 * Render a 2D tensor (matmul style).
 */
export function render2DTensor(ctx, tensor, currentIndices, cache, layouts, elementSize, tilingEnabled, tileSize) {
    const rows = tensor.rows;
    const cols = tensor.cols;

    if (cache) {
        for (let row = 0; row < rows; row++) {
            for (let col = 0; col < cols; col++) {
                if (isElementInCache2D(tensor, row, col, cache, layouts, elementSize)) {
                    ctx.fillStyle = COLORS.cached;
                    ctx.fillRect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE);
                }
            }
        }
    }

    drawGrid(ctx, rows, cols, 0, 0);

    if (tilingEnabled && tileSize > 1) {
        drawTileBoundaries(ctx, rows, cols, 0, 0, tileSize);
    }

    if (currentIndices && currentIndices.row !== undefined) {
        drawCurrentAccess(ctx, currentIndices.row, currentIndices.col, 0, 0);
    }
}

/**
 * Render a 3D tensor (channels stacked isometrically).
 */
export function render3DTensor(ctx, tensor, currentIndices, cache, layouts, elementSize, tilingEnabled, tileSize) {
    const rows = tensor.rows;
    const cols = tensor.cols;
    const channels = tensor.channels;

    const baseX = 5;
    const baseY = Math.abs((channels - 1) * ISO.depthOffsetY) + 5;

    for (let c = channels - 1; c >= 0; c--) {
        const depth = channels - 1 - c;
        const alpha = isoAlpha(depth, channels);

        const sliceOffsetX = c * ISO.depthOffsetX;
        const sliceOffsetY = c * ISO.depthOffsetY;

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

        if (cache) {
            for (let row = 0; row < rows; row++) {
                for (let col = 0; col < cols; col++) {
                    if (isElementInCache3D(tensor, c, row, col, cache, layouts, elementSize)) {
                        const pos = isoPosition(row, col, c, baseX, baseY);
                        drawCachedCell(ctx, pos.x, pos.y, alpha);
                    }
                }
            }
        }

        drawIsoGrid(ctx, rows, cols, baseX + sliceOffsetX, baseY + sliceOffsetY, alpha);

        ctx.globalAlpha = alpha;
        ctx.fillStyle = '#666';
        ctx.font = '9px sans-serif';
        ctx.fillText(`c${c}`, baseX + sliceOffsetX + 2, baseY + sliceOffsetY + rows * CELL_SIZE + 10);
        ctx.globalAlpha = 1.0;
    }

    if (tilingEnabled && tileSize > 1 && tensor.name !== 'Output') {
        drawIsoTileBoundaries(ctx, rows, cols, baseX, baseY, tileSize);
    }

    if (currentIndices && currentIndices.channel !== undefined) {
        const c = currentIndices.channel;

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

        const pos = isoPosition(currentIndices.row, currentIndices.col, c, baseX, baseY);
        drawCurrentAccessCell(ctx, pos.x, pos.y, 1.0);
    }
}

/**
 * Render a 4D tensor (kernel) as rows of isometric 3D stacks.
 */
export function render4DTensor(ctx, tensor, currentIndices, cache, layouts, elementSize) {
    const kRows = tensor.rows;
    const kCols = tensor.cols;
    const c_in = tensor.channels_in;
    const c_out = tensor.channels_out;

    const baseHeight = kRows * CELL_SIZE;
    const depthHeight = Math.abs((c_in - 1) * ISO.depthOffsetY);
    const rowHeight = baseHeight + depthHeight + CHANNEL_GAP + 5;

    const baseX = 25;

    for (let co = 0; co < c_out; co++) {
        const rowBaseY = 10 + co * rowHeight + depthHeight;

        ctx.fillStyle = '#666';
        ctx.font = '9px sans-serif';
        ctx.fillText(`co${co}`, 2, rowBaseY + kRows * CELL_SIZE / 2 + 3);

        for (let ci = c_in - 1; ci >= 0; ci--) {
            const depth = c_in - 1 - ci;
            const alpha = isoAlpha(depth, c_in);

            const sliceOffsetX = ci * ISO.depthOffsetX;
            const sliceOffsetY = ci * ISO.depthOffsetY;

            const sliceX = baseX + sliceOffsetX;
            const sliceY = rowBaseY + sliceOffsetY;

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

            if (cache) {
                for (let row = 0; row < kRows; row++) {
                    for (let col = 0; col < kCols; col++) {
                        if (isElementInCache4D(tensor, co, ci, row, col, cache, layouts, elementSize)) {
                            const cellX = sliceX + col * CELL_SIZE;
                            const cellY = sliceY + row * CELL_SIZE;
                            drawCachedCell(ctx, cellX, cellY, alpha);
                        }
                    }
                }
            }

            drawIsoGrid(ctx, kRows, kCols, sliceX, sliceY, alpha);
        }
    }

    if (currentIndices && currentIndices.c_out !== undefined) {
        const co = currentIndices.c_out;
        const ci = currentIndices.c_in;
        const rowBaseY = 10 + co * rowHeight + depthHeight;

        const sliceX = baseX + ci * ISO.depthOffsetX;
        const sliceY = rowBaseY + ci * ISO.depthOffsetY;

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

        const cellX = sliceX + currentIndices.col * CELL_SIZE;
        const cellY = sliceY + currentIndices.row * CELL_SIZE;
        drawCurrentAccessCell(ctx, cellX, cellY, 1.0);
    }
}

/**
 * Render a single tensor grid.
 * Supports 2D, 3D (with channels), and 4D (kernel) tensors.
 */
export function renderTensor(ctx, tensor, currentIndices, cache, layouts, elementSize, tilingEnabled, tileSize) {
    const canvas = ctx.canvas;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (tensor.is4D) {
        render4DTensor(ctx, tensor, currentIndices, cache, layouts, elementSize);
    } else if (tensor.is3D) {
        render3DTensor(ctx, tensor, currentIndices, cache, layouts, elementSize, tilingEnabled, tileSize);
    } else {
        render2DTensor(ctx, tensor, currentIndices, cache, layouts, elementSize, tilingEnabled, tileSize);
    }
}
