/**
 * 2D Convolution Operation Definition
 * Output[c_out][h_out][w_out] += Input[c_in][h_out+k_h][w_out+k_w] * Kernel[c_out][c_in][k_h][k_w]
 */

/**
 * Creates the 2D convolution operation.
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
export function createConv2dOperation(config = {}) {
    const {
        inputH = 8,
        inputW = 8,
        channels_in = 4,
        channels_out = 4,
        kernelH = 3,
        kernelW = 3,
        elementSize = 4
    } = config;

    const outputH = inputH - kernelH + 1;
    const outputW = inputW - kernelW + 1;

    const inputSize = inputH * inputW * channels_in;
    const kernelSize = kernelH * kernelW * channels_in * channels_out;

    const BASE_INPUT = 0;
    const BASE_KERNEL = inputSize * elementSize;
    const BASE_OUTPUT = (inputSize + kernelSize) * elementSize;

    return {
        name: 'conv2d',
        displayName: '2D Convolution',
        elementSize: elementSize,

        dimensions: {
            inputH, inputW, channels_in,
            kernelH, kernelW,
            outputH, outputW, channels_out
        },

        loopDims: ['c_out', 'h_out', 'w_out', 'c_in', 'k_h', 'k_w'],

        loopBounds: {
            c_out: channels_out,
            h_out: outputH,
            w_out: outputW,
            c_in: channels_in,
            k_h: kernelH,
            k_w: kernelW
        },

        loopOrders: {
            'c_out,h_out,w_out,c_in,k_h,k_w': ['c_out', 'h_out', 'w_out', 'c_in', 'k_h', 'k_w'],
            'h_out,w_out,c_out,c_in,k_h,k_w': ['h_out', 'w_out', 'c_out', 'c_in', 'k_h', 'k_w'],
            'c_in,k_h,k_w,c_out,h_out,w_out': ['c_in', 'k_h', 'k_w', 'c_out', 'h_out', 'w_out'],
            'k_h,k_w,c_in,c_out,h_out,w_out': ['k_h', 'k_w', 'c_in', 'c_out', 'h_out', 'w_out']
        },

        tensors: [
            {
                name: 'Input',
                baseAddress: BASE_INPUT,
                rows: inputH,
                cols: inputW,
                channels: channels_in,
                is3D: true,
                layoutOptions: [
                    { value: 'CHW', label: 'CHW' },
                    { value: 'HWC', label: 'HWC' }
                ],
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
                        return h * (inputW * channels_in) + w * channels_in + c;
                    }
                    return c * (inputH * inputW) + h * inputW + w;
                },
                getCoordinatesFromLinear: (linearIdx, layout) => {
                    if (layout === 'HWC') {
                        const h = Math.floor(linearIdx / (inputW * channels_in));
                        const w = Math.floor((linearIdx % (inputW * channels_in)) / channels_in);
                        const c = linearIdx % channels_in;
                        return { channel: c, row: h, col: w };
                    }
                    const c = Math.floor(linearIdx / (inputH * inputW));
                    const h = Math.floor((linearIdx % (inputH * inputW)) / inputW);
                    const w = linearIdx % inputW;
                    return { channel: c, row: h, col: w };
                }
            },
            {
                name: 'Kernel',
                baseAddress: BASE_KERNEL,
                rows: kernelH,
                cols: kernelW,
                channels_in: channels_in,
                channels_out: channels_out,
                is4D: true,
                layoutOptions: [
                    { value: 'OIHW', label: 'OIHW' },
                    { value: 'HWIO', label: 'HWIO' }
                ],
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
                        return h * (kernelW * channels_in * channels_out) +
                               w * (channels_in * channels_out) +
                               i * channels_out + o;
                    }
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
                rows: outputH,
                cols: outputW,
                channels: channels_out,
                is3D: true,
                layoutOptions: [
                    { value: 'CHW', label: 'CHW' },
                    { value: 'HWC', label: 'HWC' }
                ],
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
                        return h * (outputW * channels_out) + w * channels_out + c;
                    }
                    return c * (outputH * outputW) + h * outputW + w;
                },
                getCoordinatesFromLinear: (linearIdx, layout) => {
                    if (layout === 'HWC') {
                        const h = Math.floor(linearIdx / (outputW * channels_out));
                        const w = Math.floor((linearIdx % (outputW * channels_out)) / channels_out);
                        const c = linearIdx % channels_out;
                        return { channel: c, row: h, col: w };
                    }
                    const c = Math.floor(linearIdx / (outputH * outputW));
                    const h = Math.floor((linearIdx % (outputH * outputW)) / outputW);
                    const w = linearIdx % outputW;
                    return { channel: c, row: h, col: w };
                }
            }
        ],

        codeTemplate: 'Out[c_out][h][w] += In[c_in][h+kh][w+kw] * K[c_out][c_in][kh][kw]',

        describeOp: (iter) => {
            const h_in = iter.h_out + iter.k_h;
            const w_in = iter.w_out + iter.k_w;
            return `In[${iter.c_in}][${h_in}][${w_in}] × K[${iter.c_out}][${iter.c_in}][${iter.k_h}][${iter.k_w}] → Out[${iter.c_out}][${iter.h_out}][${iter.w_out}]`;
        },

        getTotalIterations: () => channels_out * outputH * outputW * channels_in * kernelH * kernelW,

        tensorOperators: ['*', '='],

        tileableDims: ['h_out', 'w_out'],
        tileSizes: [2, 4]
    };
}
