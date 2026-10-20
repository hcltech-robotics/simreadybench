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
