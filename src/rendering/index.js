/**
 * Rendering Module
 * Tensor visualization and display components
 */

export { MATRIX_SIZE, ELEMENT_SIZE, CELL_SIZE, CHANNEL_GAP, COLORS, ISO } from './config.js';

export {
    drawCell, drawCachedCell, drawCurrentAccessCell, drawParallelogram,
    isoPosition, isoAlpha, drawGrid, drawTileBoundaries,
    drawIsoGrid, drawIsoTileBoundaries, drawCurrentAccess
} from './primitives.js';

export {
    getTensorCanvasSize, renderTensor,
    render2DTensor, render3DTensor, render4DTensor,
    isElementInCache2D, isElementInCache3D, isElementInCache4D, isElementInCacheByCoords
} from './tensors.js';

export { renderTimeline, renderMemoryLayout } from './visualizations.js';
