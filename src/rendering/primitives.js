/**
 * Low-level drawing primitives for tensor visualization
 */

import { CELL_SIZE, COLORS, ISO } from './config.js';

/**
 * Draw a single cell with optional fill and border.
 */
export function drawCell(ctx, x, y, width, height, options = {}) {
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
export function drawCachedCell(ctx, x, y, alpha = 1.0) {
    drawCell(ctx, x, y, CELL_SIZE, CELL_SIZE, {
        fillColor: COLORS.cached,
        alpha: alpha
    });
}

/**
 * Draw current access indicator.
 */
export function drawCurrentAccessCell(ctx, x, y, alpha = 1.0) {
    const prevAlpha = ctx.globalAlpha;
    ctx.globalAlpha = alpha;

    ctx.fillStyle = COLORS.current;
    ctx.fillRect(x + 3, y + 3, CELL_SIZE - 6, CELL_SIZE - 6);

    ctx.strokeStyle = COLORS.currentOutline;
    ctx.lineWidth = 2;
    ctx.strokeRect(x + 1, y + 1, CELL_SIZE - 2, CELL_SIZE - 2);

    ctx.globalAlpha = prevAlpha;
}

/**
 * Draw a parallelogram (for isometric slice background/border).
 */
export function drawParallelogram(ctx, points, options = {}) {
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
 */
export function isoPosition(row, col, depth, baseX, baseY) {
    return {
        x: baseX + col * CELL_SIZE + depth * ISO.depthOffsetX,
        y: baseY + row * CELL_SIZE + depth * ISO.depthOffsetY
    };
}

/**
 * Calculate alpha for a slice at given depth.
 */
export function isoAlpha(depth, totalDepth) {
    const distanceFromFront = totalDepth - 1 - depth;
    return Math.max(0.4, ISO.sliceAlpha - distanceFromFront * ISO.backSliceAlphaDrop);
}

/**
 * Draw grid lines for a tensor slice.
 */
export function drawGrid(ctx, rows, cols, xOffset, yOffset) {
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
export function drawTileBoundaries(ctx, rows, cols, xOffset, yOffset, tileSize) {
    ctx.strokeStyle = COLORS.tileGrid;
    ctx.lineWidth = 2;

    for (let i = 0; i <= cols; i += tileSize) {
        ctx.beginPath();
        ctx.moveTo(xOffset + i * CELL_SIZE, yOffset);
        ctx.lineTo(xOffset + i * CELL_SIZE, yOffset + rows * CELL_SIZE);
        ctx.stroke();
    }
    for (let i = 0; i <= rows; i += tileSize) {
        ctx.beginPath();
        ctx.moveTo(xOffset, yOffset + i * CELL_SIZE);
        ctx.lineTo(xOffset + cols * CELL_SIZE, yOffset + i * CELL_SIZE);
        ctx.stroke();
    }
}

/**
 * Draw grid lines for an isometric slice.
 */
export function drawIsoGrid(ctx, rows, cols, xOffset, yOffset, alpha) {
    ctx.globalAlpha = alpha;
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
    ctx.globalAlpha = 1.0;
}

/**
 * Draw tile boundaries for an isometric slice.
 */
export function drawIsoTileBoundaries(ctx, rows, cols, xOffset, yOffset, tileSize) {
    ctx.strokeStyle = COLORS.tileGrid;
    ctx.lineWidth = 2;

    for (let i = 0; i <= cols; i += tileSize) {
        ctx.beginPath();
        ctx.moveTo(xOffset + i * CELL_SIZE, yOffset);
        ctx.lineTo(xOffset + i * CELL_SIZE, yOffset + rows * CELL_SIZE);
        ctx.stroke();
    }
    for (let i = 0; i <= rows; i += tileSize) {
        ctx.beginPath();
        ctx.moveTo(xOffset, yOffset + i * CELL_SIZE);
        ctx.lineTo(xOffset + cols * CELL_SIZE, yOffset + i * CELL_SIZE);
        ctx.stroke();
    }
}

/**
 * Draw current access indicator at a cell position.
 */
export function drawCurrentAccess(ctx, row, col, xOffset, yOffset) {
    const x = xOffset + col * CELL_SIZE;
    const y = yOffset + row * CELL_SIZE;

    ctx.fillStyle = COLORS.current;
    ctx.fillRect(x + 3, y + 3, CELL_SIZE - 6, CELL_SIZE - 6);
    ctx.strokeStyle = COLORS.currentOutline;
    ctx.lineWidth = 2;
    ctx.strokeRect(x + 1, y + 1, CELL_SIZE - 2, CELL_SIZE - 2);
}
