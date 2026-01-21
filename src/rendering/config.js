/**
 * Rendering configuration constants
 */

export const MATRIX_SIZE = 12;
export const ELEMENT_SIZE = 4; // bytes per element
export const CELL_SIZE = 20; // pixels per cell
export const CHANNEL_GAP = 4; // gap between channels in multi-channel visualization

export const COLORS = {
    background: '#ffffff',
    grid: '#cccccc',
    tileGrid: '#666666',
    cached: 'rgba(40, 167, 69, 0.6)',
    current: '#000000',
    currentOutline: '#667eea',
    currentSlice: '#667eea'
};

// Isometric projection configuration
export const ISO = {
    depthOffsetX: 8,
    depthOffsetY: -6,
    sliceAlpha: 0.85,
    backSliceAlphaDrop: 0.08,
    sliceGap: 2
};
