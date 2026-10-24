# Project page

The [project page](https://hcltech-robotics.github.io/simreadybench/) covers installation, local and Daytona evaluation, custom asset experiments, reinforcement-learning integration and scoring. It also presents the public reference sets, authors and model leaderboard. Its Bulma layout and publication styles use the [Academic project page template](https://github.com/eliahuhorwitz/Academic-project-page-template/tree/d38af1ccae1ce82c3404d2820c4c646afd409f81) at commit `d38af1ccae1ce82c3404d2820c4c646afd409f81`.

From the repository root, preview it with:

```sh
python -m http.server 8765 --bind 127.0.0.1 --directory site
```

Open `http://127.0.0.1:8765` to view the page. `app.js` loads leaderboard data directly from the [Hugging Face results dataset](https://huggingface.co/datasets/hcltech-robotics/simreadybench/resolve/main/leaderboard.json) through a cross-origin request. Results, scores and captures are stored on Hugging Face. The GitHub site contains the page and its presentation code.

When the dataset requires authentication, the page links to the Hugging Face leaderboard. The browser receives no embedded access token.

The protocol and evaluation-factor illustrations in `assets/` are the same figures used in the Hugging Face dataset card and GitHub README.

`assets/social-card.png` is the 1730x909 social preview used by Open Graph, Twitter, GitHub and Hugging Face previews. It contains the eyeglasses reference illustration and the project links.

The scoring section summarises the evaluation measures and links to the [scoring documentation](https://github.com/hcltech-robotics/simreadybench/blob/main/docs/evaluation.md), which contains the formal, visual and kinematic fidelity charts.

The template's `bulma.min.css` and `index.css` are in `static/css/`. Benchmark typography, table and command styles are in `style.css`. The footer and [template licence](TEMPLATE-LICENSE.md) provide the required attribution.

## Interactive object comparison

The `#explorer` section compares five NV25 reference objects with recorded model reconstructions. `explorer.js` reads `assets/explorer/manifest.json` and loads the local `vendor/model-viewer.min.js` module when the section approaches the viewport. Both models can be rotated with a pointer, touch or the labelled rotation buttons. Linked rotation preserves each model's own framing; it does not claim matched scale. Each object and model change replaces its viewer element so an older asynchronous load cannot appear alongside a new trial's scores.

The manifest associates each geometry preview with its original asset, run, zero-based repetition, source link and checksum. Scores and usage belong to that exact reconstruction, not a leaderboard average. The displayed trial number is the recorded repetition plus one. Missing measurements stay missing, a measured zero stays zero, and inapplicable kinematic criteria show `N/A`. Total cost is shown only when recorded; otherwise an available inference cost is explicitly labelled. Inference costs derived from pricing retain that qualifier. Tokens include the recorded input/output breakdown and time is elapsed time for the reconstruction, with model-request time separately identified. Orichalcum conditions are attributed only by their pin.

The browser previews are geometry exports with neutral materials. Formal, visual and kinematic measurements remain those from evaluation of the original USD packages. The preview does not run physics or reproduce the evaluated material rendering. The section uses the site's centred heading, Bulma controls and result-table typography. Compact labels retain the selected trial, neutral-material/independent-scale caveat and reference licence; the selection policy and conversion details are documented in the asset manifest and its README.

The Pages workflow previews with `?snapshot=1`, which leaves the object viewer dormant while serialising the live leaderboard. This prevents transient viewer and loading state from being saved into the published HTML. Normal visitors initialise the viewer on demand. Geometry or WebGL failures leave the measured readout and source links available; a reconstruction with `src: null` gets its own unavailable preview without disabling other selections. A manifest or library failure provides the Hugging Face fallback.

With the local preview server running and Playwright/Chromium installed, verify every object/run selection, linked rotation, mobile layout and snapshot isolation:

```sh
python3 scripts/check_viewer_browser.py http://127.0.0.1:8765 --output /tmp/simreadybench-viewer-check
```

The check writes desktop/mobile screenshots and a verification receipt. Geometry export provenance, attribution and data rebuild instructions are in [assets/explorer/README.md](assets/explorer/README.md).

## Search indexing

The page serves its benchmark description, methods and documentation links in HTML. JavaScript adds the live leaderboard, with a Hugging Face link available before it loads. The HTML head contains the canonical URL, matching search and social descriptions, and linked `WebPage`/`SoftwareSourceCode` metadata.

After deployment, submit `https://hcltech-robotics.github.io/simreadybench/sitemap.xml` in Google Search Console for the `https://hcltech-robotics.github.io/simreadybench/` URL-prefix property and request indexing of the homepage. The sitemap contains the canonical page URL. Section anchors are parts of that page.

GitHub Pages hosts this project under `/simreadybench/`. Crawlers read `robots.txt` at `https://hcltech-robotics.github.io/robots.txt`, which belongs to the organisation's root site. A `robots.txt` in this directory would not control crawling. If the organisation maintains a root robots file, it can advertise this sitemap there.

When moving the site, update the canonical URL, social URLs, JSON-LD identifiers and sitemap together. Keep the software version consistent with `pyproject.toml` and add publication metadata when the paper is available.
