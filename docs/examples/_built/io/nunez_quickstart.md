# Load and plot a Nunez-Elizalde recording

Fetch a small subset of the Nunez-Elizalde 2022 dataset, load a power Doppler time
series, and visualize the mean volume in dB.

```python
from pathlib import Path

import confusius as cf
from confusius.datasets import fetch_nunez_elizalde_2022

# Download dataset (cached after the first run, ~30 MB).
bids_root = fetch_nunez_elizalde_2022(
    subjects="CR022",
    sessions="20201011",
    tasks="spontaneous",
    acqs="slice03",
)

# Load power Doppler time series.
pwd_path = (
    Path(bids_root)
    / "sub-CR022"
    / "ses-20201011"
    / "fusi"
    / "sub-CR022_ses-20201011_task-spontaneous_acq-slice03_pwd.nii.gz"
)
data = cf.load(pwd_path)
data
```

<div class="gallery-rich-output jupyter_cell jupyter_container docutils container"><div class="cell_output docutils container"><div class="output text_html"><div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in notebooks */

:root {
  --xr-font-color0: var(
    --jp-content-font-color0,
    var(--pst-color-text-base rgba(0, 0, 0, 1))
  );
  --xr-font-color2: var(
    --jp-content-font-color2,
    var(--pst-color-text-base, rgba(0, 0, 0, 0.54))
  );
  --xr-font-color3: var(
    --jp-content-font-color3,
    var(--pst-color-text-base, rgba(0, 0, 0, 0.38))
  );
  --xr-border-color: var(
    --jp-border-color2,
    hsl(from var(--pst-color-on-background, white) h s calc(l - 10))
  );
  --xr-disabled-color: var(
    --jp-layout-color3,
    hsl(from var(--pst-color-on-background, white) h s calc(l - 40))
  );
  --xr-background-color: var(
    --jp-layout-color0,
    var(--pst-color-on-background, white)
  );
  --xr-background-color-row-even: var(
    --jp-layout-color1,
    hsl(from var(--pst-color-on-background, white) h s calc(l - 5))
  );
  --xr-background-color-row-odd: var(
    --jp-layout-color2,
    hsl(from var(--pst-color-on-background, white) h s calc(l - 15))
  );
}

html[theme="dark"],
html[data-theme="dark"],
body[data-theme="dark"],
body.vscode-dark {
  --xr-font-color0: var(
    --jp-content-font-color0,
    var(--pst-color-text-base, rgba(255, 255, 255, 1))
  );
  --xr-font-color2: var(
    --jp-content-font-color2,
    var(--pst-color-text-base, rgba(255, 255, 255, 0.54))
  );
  --xr-font-color3: var(
    --jp-content-font-color3,
    var(--pst-color-text-base, rgba(255, 255, 255, 0.38))
  );
  --xr-border-color: var(
    --jp-border-color2,
    hsl(from var(--pst-color-on-background, #111111) h s calc(l + 10))
  );
  --xr-disabled-color: var(
    --jp-layout-color3,
    hsl(from var(--pst-color-on-background, #111111) h s calc(l + 40))
  );
  --xr-background-color: var(
    --jp-layout-color0,
    var(--pst-color-on-background, #111111)
  );
  --xr-background-color-row-even: var(
    --jp-layout-color1,
    hsl(from var(--pst-color-on-background, #111111) h s calc(l + 5))
  );
  --xr-background-color-row-odd: var(
    --jp-layout-color2,
    hsl(from var(--pst-color-on-background, #111111) h s calc(l + 15))
  );
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
  line-height: 1.6;
  padding-bottom: 4px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
}

.xr-header {
  border-bottom: solid 1px var(--xr-border-color);
  margin-bottom: 4px;
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-obj-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type,
.xr-group-box-contents > label {
  color: var(--xr-font-color2);
  display: block;
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 0 20px 0 20px;
  margin-block-start: 0;
  margin-block-end: 0;
}

.xr-section-item {
  display: contents;
}

.xr-section-item > input,
.xr-group-box-contents > input,
.xr-array-wrap > input {
  display: block;
  opacity: 0;
  height: 0;
  margin: 0;
}

.xr-section-item > input + label,
.xr-var-item > input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item > input:enabled + label,
.xr-var-item > input:enabled + label,
.xr-array-wrap > input:enabled + label,
.xr-group-box-contents > input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item > input:focus-visible + label,
.xr-var-item > input:focus-visible + label,
.xr-array-wrap > input:focus-visible + label,
.xr-group-box-contents > input:focus-visible + label {
  outline: auto;
}

.xr-section-item > input:enabled + label:hover,
.xr-var-item > input:enabled + label:hover,
.xr-array-wrap > input:enabled + label:hover,
.xr-group-box-contents > input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
  white-space: nowrap;
}

.xr-section-summary > em {
  font-weight: normal;
}

.xr-span-grid {
  grid-column-end: -1;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.3em;
}

.xr-group-box-contents > input:checked + label > span {
  display: inline-block;
  padding-left: 0.6em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: "►";
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: "▼";
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details,
.xr-group-box-contents > label {
  padding-top: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  grid-column: 1 / -1;
  margin-top: 4px;
  margin-bottom: 5px;
}

.xr-section-summary-in ~ .xr-section-details {
  display: none;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-children {
  display: inline-grid;
  grid-template-columns: 100%;
  grid-column: 1 / -1;
  padding-top: 4px;
}

.xr-group-box {
  display: inline-grid;
  grid-template-columns: 0px 30px auto;
}

.xr-group-box-vline {
  grid-column-start: 1;
  border-right: 0.2em solid;
  border-color: var(--xr-border-color);
  width: 0px;
}

.xr-group-box-hline {
  grid-column-start: 2;
  grid-row-start: 1;
  height: 1em;
  width: 26px;
  border-bottom: 0.2em solid;
  border-color: var(--xr-border-color);
}

.xr-group-box-contents {
  grid-column-start: 3;
  padding-bottom: 4px;
}

.xr-group-box-contents > label::before {
  content: "📂";
  padding-right: 0.3em;
}

.xr-group-box-contents > input:checked + label::before {
  content: "📁";
}

.xr-group-box-contents > input:checked + label {
  padding-bottom: 0px;
}

.xr-group-box-contents > input:checked ~ .xr-sections {
  display: none;
}

.xr-group-box-contents > input + label > span {
  display: none;
}

.xr-group-box-ellipsis {
  font-size: 1.4em;
  font-weight: 900;
  color: var(--xr-font-color2);
  letter-spacing: 0.15em;
  cursor: default;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: "(";
}

.xr-dim-list:after {
  content: ")";
}

.xr-dim-list li:not(:last-child):after {
  content: ",";
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  border-color: var(--xr-background-color-row-odd);
  margin-bottom: 0;
  padding-top: 2px;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
  border-color: var(--xr-background-color-row-even);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  border-top: 2px dotted var(--xr-background-color);
  padding-bottom: 20px !important;
  padding-top: 10px !important;
}

.xr-var-attrs-in + label,
.xr-var-data-in + label,
.xr-index-data-in + label {
  padding: 0 1px;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-data > pre,
.xr-index-data > pre,
.xr-var-data > table > tbody > tr {
  background-color: transparent !important;
}

.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}

.xr-var-attrs-in:checked + label > .xr-icon-file-text2,
.xr-var-data-in:checked + label > .xr-icon-database,
.xr-index-data-in:checked + label > .xr-icon-database {
  color: var(--xr-font-color0);
  filter: drop-shadow(1px 1px 5px var(--xr-font-color2));
  stroke-width: 0.8px;
}
</style><div class='xr-wrap'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-obj-name'>&#x27;sub-CR022_ses-20201011_task-spontaneous_acq-slice03_pwd&#x27;</div><ul class='xr-dim-list'><li><span class='xr-has-index'>time</span>: 751</li><li><span class='xr-has-index'>z</span>: 1</li><li><span class='xr-has-index'>y</span>: 114</li><li><span class='xr-has-index'>x</span>: 80</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-fbb8ad79-e952-43f1-b206-4030799c1054' class='xr-array-in' type='checkbox' checked><label for='section-fbb8ad79-e952-43f1-b206-4030799c1054' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>dask.array&lt;chunksize=(751, 1, 114, 80), meta=np.ndarray&gt;</span></div><div class='xr-array-data'><table>
    <tr>
        <td>
            <table style="border-collapse: collapse;">
                <thead>
                    <tr>
                        <td> </td>
                        <th> Array </th>
                        <th> Chunk </th>
                    </tr>
                </thead>
                <tbody>
                    
                    <tr>
                        <th> Bytes </th>
                        <td> 26.13 MiB </td>
                        <td> 26.13 MiB </td>
                    </tr>
                    
                    <tr>
                        <th> Shape </th>
                        <td> (751, 1, 114, 80) </td>
                        <td> (751, 1, 114, 80) </td>
                    </tr>
                    <tr>
                        <th> Dask graph </th>
                        <td colspan="2"> 1 chunks in 1 graph layer </td>
                    </tr>
                    <tr>
                        <th> Data type </th>
                        <td colspan="2"> float32 numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="483" height="105" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="25" x2="120" y2="25" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="25" style="stroke-width:2" />
  <line x1="120" y1="0" x2="120" y2="25" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,25.412616514582485 0.0,25.412616514582485" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.0" y="45.41261651458248" font-size="1.0rem" font-weight="100" text-anchor="middle" >751</text>
  <text x="140.0" y="12.706308257291242" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.0,12.706308257291242)">1</text>


  <!-- Horizontal lines -->
  <line x1="190" y1="0" x2="204" y2="14" style="stroke-width:2" />
  <line x1="190" y1="40" x2="204" y2="55" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="190" y1="0" x2="190" y2="40" style="stroke-width:2" />
  <line x1="204" y1="14" x2="204" y2="55" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="190.0,0.0 204.9485979497544,14.948597949754403 204.9485979497544,55.712341781689695 190.0,40.763743831935294" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Horizontal lines -->
  <line x1="190" y1="0" x2="228" y2="0" style="stroke-width:2" />
  <line x1="204" y1="14" x2="243" y2="14" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="190" y1="0" x2="204" y2="14" style="stroke-width:2" />
  <line x1="228" y1="0" x2="243" y2="14" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="190.0,0.0 228.94414111279977,0.0 243.89273906255417,14.948597949754403 204.9485979497544,14.948597949754403" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Horizontal lines -->
  <line x1="204" y1="14" x2="243" y2="14" style="stroke-width:2" />
  <line x1="204" y1="55" x2="243" y2="55" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="204" y1="14" x2="204" y2="55" style="stroke-width:2" />
  <line x1="243" y1="14" x2="243" y2="55" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="204.9485979497544,14.948597949754403 243.89273906255417,14.948597949754403 243.89273906255417,55.712341781689695 204.9485979497544,55.712341781689695" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="224.4206685061543" y="75.7123417816897" font-size="1.0rem" font-weight="100" text-anchor="middle" >80</text>
  <text x="263.89273906255414" y="35.33046986572205" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(-90,263.89273906255414,35.33046986572205)">114</text>
  <text x="187.4742989748772" y="68.2380428068125" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(45,187.4742989748772,68.2380428068125)">1</text>
</svg>
        </td>
    </tr>
</table></div></div></li><li class='xr-section-item'><input id='section-3ddafc52-e9ab-4147-aa61-9055930d50ad' class='xr-section-summary-in' type='checkbox' checked /><label for='section-3ddafc52-e9ab-4147-aa61-9055930d50ad' class='xr-section-summary' title='Expand/collapse section'>Coordinates: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>time</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>10.61 10.91 11.21 ... 235.4 235.7</div><input id='attrs-858a8698-a5c9-4afa-bc23-982872e48102' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-858a8698-a5c9-4afa-bc23-982872e48102' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d2d8ce30-c94c-46cb-804e-a66b0114ac05' class='xr-var-data-in' type='checkbox'><label for='data-d2d8ce30-c94c-46cb-804e-a66b0114ac05' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>units :</span></dt><dd>s</dd><dt><span>volume_acquisition_reference :</span></dt><dd>start</dd><dt><span>volume_acquisition_duration :</span></dt><dd>0.3</dd></dl></div><div class='xr-var-data'><pre>array([ 10.608,  10.908,  11.208, ..., 235.095, 235.395, 235.695], shape=(751,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>z</span></div><div class='xr-var-dims'>(z)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.0</div><input id='attrs-09a90765-58d4-471e-94de-e9aa77746ac7' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-09a90765-58d4-471e-94de-e9aa77746ac7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-67238c85-0908-4883-8313-bee3accd07d6' class='xr-var-data-in' type='checkbox'><label for='data-67238c85-0908-4883-8313-bee3accd07d6' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>units :</span></dt><dd>mm</dd><dt><span>voxdim :</span></dt><dd>0.4000000059604645</dd></dl></div><div class='xr-var-data'><pre>array([1.])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>y</span></div><div class='xr-var-dims'>(y)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>2.73 2.778 2.827 ... 8.142 8.19</div><input id='attrs-95ce848e-9b71-4bdb-9c60-e75f8e6a7618' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-95ce848e-9b71-4bdb-9c60-e75f8e6a7618' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9c7eba43-3b66-41d6-b797-ac18c1cb3bf4' class='xr-var-data-in' type='checkbox'><label for='data-9c7eba43-3b66-41d6-b797-ac18c1cb3bf4' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>units :</span></dt><dd>mm</dd><dt><span>voxdim :</span></dt><dd>0.04831999912858009</dd></dl></div><div class='xr-var-data'><pre>array([2.73008, 2.7784 , 2.82672, 2.87504, 2.92336, 2.97168, 3.02   , 3.06832,
       3.11664, 3.16496, 3.21328, 3.2616 , 3.30992, 3.35824, 3.40656, 3.45488,
       3.5032 , 3.55152, 3.59984, 3.64816, 3.69648, 3.7448 , 3.79312, 3.84144,
       3.88976, 3.93808, 3.9864 , 4.03472, 4.08304, 4.13136, 4.17968, 4.228  ,
       4.27632, 4.32464, 4.37296, 4.42128, 4.4696 , 4.51792, 4.56624, 4.61456,
       4.66288, 4.7112 , 4.75952, 4.80784, 4.85616, 4.90448, 4.9528 , 5.00112,
       5.04944, 5.09776, 5.14608, 5.1944 , 5.24272, 5.29104, 5.33936, 5.38768,
       5.436  , 5.48432, 5.53264, 5.58096, 5.62928, 5.6776 , 5.72592, 5.77424,
       5.82256, 5.87088, 5.9192 , 5.96752, 6.01584, 6.06416, 6.11248, 6.1608 ,
       6.20912, 6.25744, 6.30576, 6.35408, 6.4024 , 6.45072, 6.49904, 6.54736,
       6.59568, 6.644  , 6.69232, 6.74064, 6.78896, 6.83728, 6.8856 , 6.93392,
       6.98224, 7.03056, 7.07888, 7.1272 , 7.17552, 7.22384, 7.27216, 7.32048,
       7.3688 , 7.41712, 7.46544, 7.51376, 7.56208, 7.6104 , 7.65872, 7.70704,
       7.75536, 7.80368, 7.852  , 7.90032, 7.94864, 7.99696, 8.04528, 8.0936 ,
       8.14192, 8.19024])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>x</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-3.95 -3.85 -3.75 ... 3.85 3.95</div><input id='attrs-f4580e79-a53d-41bc-bed9-e33df9ca32a3' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-f4580e79-a53d-41bc-bed9-e33df9ca32a3' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-8219d4cc-f853-4245-8ec3-0708daac3197' class='xr-var-data-in' type='checkbox'><label for='data-8219d4cc-f853-4245-8ec3-0708daac3197' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>units :</span></dt><dd>mm</dd><dt><span>voxdim :</span></dt><dd>0.10000000149011612</dd></dl></div><div class='xr-var-data'><pre>array([-3.95, -3.85, -3.75, -3.65, -3.55, -3.45, -3.35, -3.25, -3.15, -3.05,
       -2.95, -2.85, -2.75, -2.65, -2.55, -2.45, -2.35, -2.25, -2.15, -2.05,
       -1.95, -1.85, -1.75, -1.65, -1.55, -1.45, -1.35, -1.25, -1.15, -1.05,
       -0.95, -0.85, -0.75, -0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05,
        0.05,  0.15,  0.25,  0.35,  0.45,  0.55,  0.65,  0.75,  0.85,  0.95,
        1.05,  1.15,  1.25,  1.35,  1.45,  1.55,  1.65,  1.75,  1.85,  1.95,
        2.05,  2.15,  2.25,  2.35,  2.45,  2.55,  2.65,  2.75,  2.85,  2.95,
        3.05,  3.15,  3.25,  3.35,  3.45,  3.55,  3.65,  3.75,  3.85,  3.95])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-f47c53da-1f3c-4113-8cfe-6680ea48ba58' class='xr-section-summary-in' type='checkbox' /><label for='section-f47c53da-1f3c-4113-8cfe-6680ea48ba58' class='xr-section-summary' title='Expand/collapse section'>Attributes: <span>(25)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>qform_code :</span></dt><dd>1</dd><dt><span>manufacturer :</span></dt><dd>Verasonics</dd><dt><span>manufacturers_model_name :</span></dt><dd>Vantage 128</dd><dt><span>software_version :</span></dt><dd>Alan Urban Technology &amp; Consulting (AUTC)</dd><dt><span>probe_manufacturer :</span></dt><dd>Vermon</dd><dt><span>probe_type :</span></dt><dd>linear</dd><dt><span>probe_model :</span></dt><dd>L22-XTech</dd><dt><span>probe_central_frequency :</span></dt><dd>15000000.0</dd><dt><span>probe_number_of_elements :</span></dt><dd>128</dd><dt><span>probe_pitch :</span></dt><dd>0.1</dd><dt><span>probe_focal_width :</span></dt><dd>0.4</dd><dt><span>probe_focal_depth :</span></dt><dd>8.0</dd><dt><span>power_doppler_integration_duration :</span></dt><dd>0.3</dd><dt><span>power_doppler_integration_stride :</span></dt><dd>0.3</dd><dt><span>clutter_filter_window_duration :</span></dt><dd>0.4</dd><dt><span>clutter_filter_window_stride :</span></dt><dd>0.3</dd><dt><span>clutter_filters :</span></dt><dd>[&#x27;highpass:15Hz&#x27;, &#x27;svd:remove_first_15_components&#x27;]</dd><dt><span>task_name :</span></dt><dd>spontaneous</dd><dt><span>task_description :</span></dt><dd>Spontaneous activity without explicit visual stimulation.</dd><dt><span>depth :</span></dt><dd>[0.0, 5.46016]</dd><dt><span>transmit_frequency :</span></dt><dd>15625000.0</dd><dt><span>compound_sampling_frequency :</span></dt><dd>500.0</dd><dt><span>plane_wave_angles :</span></dt><dd>[-10.0, -7.9, -5.8, -3.6999999999999993, -1.5999999999999996, 0.5000000000000018, 2.6000000000000014, 4.700000000000002, 6.8000000000000025, 8.900000000000002]</dd><dt><span>probe_voltage :</span></dt><dd>25.0</dd><dt><span>affines :</span></dt><dd>{&#x27;physical_to_qform&#x27;: array([[1., 0., 0., 0.],
       [0., 1., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.]])}</dd></dl></div></li></ul></div></div>
<style>.xr-text-repr-fallback{display:none!important;}.xr-array-preview,.xr-array-preview span,.xr-preview,.xr-var-preview,.xr-var-dtype,.xr-var-dims,.xr-var-name,.xr-obj-type,.xr-obj-name{color:var(--xr-font-color0)!important;}.gallery-rich-output{overflow-x:auto;}.gallery-rich-output .xr-var-list,.gallery-rich-output .xr-dim-list,.gallery-rich-output .xr-attrs{padding-left:0!important;margin:0!important;list-style:none!important;}.gallery-rich-output .xr-var-list,.gallery-rich-output .xr-var-item{display:contents!important;}.gallery-rich-output .xr-section-summary-in~.xr-section-details{display:none!important;}.gallery-rich-output .xr-section-summary-in:checked~.xr-section-details{display:contents!important;}.gallery-rich-output .xr-var-name,.gallery-rich-output .xr-var-dims,.gallery-rich-output .xr-var-dtype,.gallery-rich-output .xr-var-preview{margin:0!important;}.gallery-rich-output .xr-array-data svg text,.gallery-rich-output .xr-array-data svg tspan,.gallery-rich-output .xr-array-preview svg text,.gallery-rich-output .xr-array-preview svg tspan,.gallery-rich-output .xr-preview svg text,.gallery-rich-output .xr-preview svg tspan{fill:var(--xr-font-color0)!important;}.gallery-rich-output .xr-array-data svg{color:var(--xr-font-color0)!important;}[data-md-color-scheme='default'] .xr-wrap{--xr-font-color0: rgba(0,0,0,1);--xr-font-color2: rgba(0,0,0,0.62);--xr-font-color3: rgba(0,0,0,0.42);--xr-border-color: var(--md-default-bg-color--lightest);--xr-disabled-color: rgba(0,0,0,0.35);--xr-background-color: #fff;--xr-background-color-row-even: #f8f9fb;--xr-background-color-row-odd: #edf0f5;}[data-md-color-scheme='slate'] .xr-wrap{--xr-font-color0: rgba(255,255,255,0.95);--xr-font-color2: rgba(255,255,255,0.68);--xr-font-color3: rgba(255,255,255,0.45);--xr-border-color: #2a3347;--xr-disabled-color: rgba(255,255,255,0.28);--xr-background-color: #111720;--xr-background-color-row-even: #161d29;--xr-background-color-row-odd: #1d2533;}</style></div></div></div>

Average over time and convert to dB scale for a quick static preview.

```python
import matplotlib.pyplot as plt

mean_db = data.mean("time").fusi.scale.db()

# Plot all z-slices.
plotter = mean_db.fusi.plot.volume(
    cmap="gray",
    cbar_label="Power Doppler (dB)",
)
plt.show()
```

<img class="skip-lightbox" src="nunez_quickstart_output_light/cell_04_0_light.png#only-light" alt="Example output from cell 4, image 0"><img class="skip-lightbox" src="nunez_quickstart_output_dark/cell_04_0_dark.png#only-dark" alt="Example output from cell 4, image 0">


---

**Total running time:** 6.0 s


[Download .py](nunez_quickstart.py){ .md-button } [Download .ipynb](nunez_quickstart.ipynb){ .md-button }
