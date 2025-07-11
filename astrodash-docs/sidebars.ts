import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  apiSidebar: [
    'api/swagger',
    {
      type: 'doc',
      id: 'api/intro',
      label: 'Introduction',
    },
    {
      type: 'category',
      label: 'Endpoints',
      items: [
        'api/endpoints/health',
        'api/endpoints/process-spectrum',
        'api/endpoints/batch-process',
        'api/endpoints/analysis-options',
        'api/endpoints/template-spectrum',
        'api/endpoints/estimate-redshift',
        // 'api/endpoints/line-list',
        'api/endpoints/osc-references',
      ],
    },
  ],
  guidesSidebar: [
    {
      type: 'doc',
      id: 'guides/getting-started',
      label: 'Getting Started',
    },
    {
      type: 'category',
      label: 'Code Examples',
      items: [
        'guides/code-examples/python',
      ],
    },
  ],
};

export default sidebars;
