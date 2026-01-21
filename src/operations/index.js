/**
 * Operations Module
 * Registry and exports for tensor operation definitions
 */

import { createMatmulOperation } from './matmul.js';
import { createConv2dOperation } from './conv2d.js';
import { MATRIX_SIZE, ELEMENT_SIZE } from '../rendering/config.js';

// Operation registry
export const OPERATIONS = {
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

// Re-export operation creators
export { createMatmulOperation } from './matmul.js';
export { createConv2dOperation } from './conv2d.js';
