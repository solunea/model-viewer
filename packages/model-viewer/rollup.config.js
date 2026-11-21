/*
 * Copyright 2019 Google LLC. All Rights Reserved.
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
import {rollup} from 'rollup';

const {NODE_ENV} = process.env;

const onwarn = (warning, warn) => {
  // Suppress non-actionable warning caused by TypeScript boilerplate:
  if (warning.code !== 'THIS_IS_UNDEFINED') {
    warn(warning);
  }
};

const depthWorkerEntry = './lib/three-components/SkyboxDepthWorker.js';
const depthWorker = {
  name: 'inline-skybox-depth-worker',
  async transform(code, id) {
    if (!id.endsWith('/SkyboxDepth.js') && !id.endsWith('\\SkyboxDepth.js')) {
      return null;
    }
    this.addWatchFile(depthWorkerEntry);
    const build = await rollup({input: depthWorkerEntry, plugins: [resolve(), terser()]});
    try {
      const {output} = await build.generate({format: 'iife'});
      const source = JSON.stringify(output[0].code);
      const factory =
          "return new Worker(new URL('./SkyboxDepthWorker.js', import.meta.url), { type: 'module' });";
      if (!code.includes(factory)) {
        this.error('Could not embed the panorama depth worker.');
      }
      return {
        code: code.replace(
            factory,
            `const url = URL.createObjectURL(new Blob([${source}], {type: 'text/javascript'}));
             try {
               return new Worker(url, {type: 'module'});
             } finally {
               URL.revokeObjectURL(url);
             }`),
        map: null
      };
    } finally {
      await build.close();
    }
  },
};

let commonPlugins = [
  resolve({dedupe: 'three'}),
  depthWorker,
  replace({'Reflect.decorate': 'undefined', preventAssignment: true})
];

const watchFiles = ['lib/**'];

const createModelViewerOutput =
    (file, format, plugins = commonPlugins, external = []) => {
      const globals = external.reduce((acc, mod) => {
        acc[mod] =
            mod;  // Assuming global variable names are the same as module names
        return acc;
      }, {});

      return {
        input: './lib/model-viewer.js',
        output: {
          file,
          format,
          sourcemap: true,
          name: 'ModelViewerElement',
          globals
        },
        external,
        watch: {include: watchFiles},
        plugins,
        onwarn
      };
    };

const outputOptions = [
  createModelViewerOutput('./dist/model-viewer.js', 'esm'),
  createModelViewerOutput(
      './dist/model-viewer-module.js', 'esm', commonPlugins, ['three'])
];

if (NODE_ENV !== 'development') {
  const pluginsIE11 = [
    ...commonPlugins,
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
  outputOptions.push(
      createModelViewerOutput('./dist/model-viewer-umd.js', 'umd', pluginsIE11),
      /** Bundled w/o three */
      createModelViewerOutput(
          './dist/model-viewer-module-umd.js', 'umd', pluginsIE11, ['three']));

  // Minified Versions
  const minifiedPlugins = [...commonPlugins, terser()];

  outputOptions.push(
      createModelViewerOutput(
          './dist/model-viewer.min.js', 'esm', minifiedPlugins),
      createModelViewerOutput(
          './dist/model-viewer-umd.min.js', 'umd', minifiedPlugins),
      createModelViewerOutput(
          './dist/model-viewer-module.min.js',
          'esm',
          minifiedPlugins,
          ['three']),
      createModelViewerOutput(
          './dist/model-viewer-module-umd.min.js', 'umd', minifiedPlugins, [
            'three'
          ]));

  outputOptions.push({
    input: './lib/model-viewer.d.ts',
    output: {
      file: './dist/model-viewer.d.ts',
      format: 'esm',
      name: 'ModelViewerElement',
    },
    plugins: [dts()],
  });
}

export default outputOptions;
