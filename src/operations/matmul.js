/**
 * Matrix Multiplication Operation Definition
 * C[i][j] += A[i][k] * B[k][j]
 */

/**
 * Creates the matrix multiplication operation.
 *
 * @param {number} size - Matrix dimension (e.g., 12)
 * @param {number} elementSize - Bytes per element (e.g., 4)
 * @returns {Object} Operation definition
 */
export function createMatmulOperation(size, elementSize) {
    const elementsPerMatrix = size * size;

    return {
        name: 'matmul',
        displayName: 'Matrix Multiplication',
        size: size,
        elementSize: elementSize,

        loopDims: ['i', 'j', 'k'],

        loopBounds: { i: size, j: size, k: size },

        loopOrders: {
            'ijk': ['i', 'j', 'k'],
            'ikj': ['i', 'k', 'j'],
            'jik': ['j', 'i', 'k'],
            'jki': ['j', 'k', 'i'],
            'kij': ['k', 'i', 'j'],
            'kji': ['k', 'j', 'i']
        },

        tensors: [
            {
                name: 'A',
                baseAddress: 0,
                rows: size,
                cols: size,
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

        codeTemplate: 'C[i][j] += A[i][k] * B[k][j]',

        describeOp: (iter) => `A[${iter.i}][${iter.k}] × B[${iter.k}][${iter.j}] → C[${iter.i}][${iter.j}]`,

        getTotalIterations: () => size * size * size,

        tensorOperators: ['×', '='],

        tileableDims: ['i', 'j', 'k'],
        tileSizes: [2, 4, 6]
    };
}
