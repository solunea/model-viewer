/*
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import commonjs from '@rollup/plugin-commonjs';
import {nodeResolve as resolve} from '@rollup/plugin-node-resolve';
import replace from '@rollup/plugin-replace';
import swc from '@rollup/plugin-swc';
import terser from '@rollup/plugin-terser';
import cleanup from 'rollup-plugin-cleanup';
import dts from 'rollup-plugin-dts';

const {NODE_ENV} = process.env;

const onwarn = (warning, warn) => {
  // Suppress non-actionable warning caused by TypeScript boilerplate:
  if (warning.code !== 'THIS_IS_UNDEFINED') {
    warn(warning);
  }
};

const plugins = [
  resolve(),
  replace({'Reflect.decorate': 'undefined', preventAssignment: true})
];

const watchFiles = ['lib/**'];
const effectEntry = './lib/model-viewer-effects.js';

const gpuCompanion = minified => ({
  name: 'model-viewer-effects-webgpu-companion',
  resolveId(id, importer) {
    if (id === './WebGPUEffects.js' && importer?.endsWith('effect-composer.js')) {
      return {
        id: minified ? './model-viewer-effects-webgpu.min.js' :
                      './model-viewer-effects-webgpu.js',
        external: true
      };
    }
    return null;
  }
});

function output(file, format, plugins, minified = false, gpu = false) {
  return {
    input: gpu ? './lib/WebGPUEffects.js' : effectEntry,
    output: {
      file: './dist/' + file,
      sourcemap: true,
      format,
      name: 'ModelViewerEffects',
      globals: {three: 'three'}
    },
    watch: {include: watchFiles},
    plugins: gpu ? plugins : [gpuCompanion(minified), ...plugins],
    external: gpu ? ['three', 'three/webgpu', 'three/tsl'] : ['three'],
    onwarn
  };
}

const outputOptions = [
  output('model-viewer-effects.js', 'esm', plugins),
  output('model-viewer-effects-webgpu.js', 'esm', plugins, false, true)
];

if (NODE_ENV !== 'development') {
  const pluginsIE11 = [
    ...plugins,
    commonjs(),
    swc(),
    cleanup({
      // Ideally we'd also clean third_party/three, which saves
      // ~45kb in filesize alone... but takes 2 minutes to build
      include: ['lib/**'],
      comments: 'none',
    }),
  ];

  // IE11 does not support modules, so they are removed here, as well as in a
  // dedicated unit test build which is needed for the same reason.
  const minified = [...plugins, terser()];
  outputOptions.push(
      output('model-viewer-effects-umd.js', 'umd', pluginsIE11),
      output('model-viewer-effects.min.js', 'esm', minified, true),
      output('model-viewer-effects-webgpu.min.js', 'esm', minified, true, true),
      output('model-viewer-effects-umd.min.js', 'umd',
             [...pluginsIE11, terser()], true));

  outputOptions.push({
    input: './lib/model-viewer-effects.d.ts',
    output: {
      file: './dist/model-viewer-effects.d.ts',
      format: 'esm',
      name: 'ModelViewerEffects',
    },
    plugins: [dts()],
    external: ['three', 'three/webgpu', 'three/tsl'],
  });
}

export default outputOptions;
