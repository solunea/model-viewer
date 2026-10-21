import {playwrightLauncher} from '@web/test-runner-playwright';
import base from './web-test-runner.config.mjs';

export default {
  ...base,
  files: 'lib/test/features/panorama-areas-spec.js',
  concurrency: 1,
  testFramework: {config: {ui: 'tdd', timeout: 30000}},
  browsers: ['chromium', 'firefox', 'webkit'].map(product => playwrightLauncher({
    product,
    launchOptions: product === 'chromium' ? {args: ['--enable-unsafe-swiftshader']} : {}
  }))
};
