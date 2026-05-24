# Ecosystem integration

This page describes how a repo of the PyADM1ODE ecosystem hooks into the
others. It also serves as the migration guide for bringing the two sister
repos (`PyADM1ODE`, `PyADM1ODE_calibration`) up to the same standard as
`PyADM1ODE_estimate`.

## Overview — the five tactics

| # | Tactic | Effort |
|---|---|---|
| 1 | Same Material theme + palette | < 5 min |
| 2 | `objects.inv` cross-references | < 10 min |
| 3 | Consistent top navigation (Ecosystem section) | < 10 min |
| 4 | Footer with eco-branding | 2 min |
| 5 | Keep `mike` versions in sync | ongoing, organizational |

Concrete snippets below.

## Tactic 1 — Align theme palette

All three repos should share the same Material configuration in `mkdocs.yml`:

```yaml
theme:
  name: material
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.expand
    - navigation.top
    - navigation.tracking
    - search.suggest
    - search.highlight
    - search.share
    - content.code.copy
    - content.code.annotate
    - content.tooltips
  palette:
    - scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  font:
    text: Roboto
    code: Roboto Mono
```

The `language` setting per repo stays unchanged (`de` as default,
the i18n plugin renders the switcher).

## Tactic 2 — Cross-API references via `objects.inv`

`mkdocstrings` can resolve references to Python objects in another doc site
as soon as that site's `objects.inv` is published. In `mkdocs.yml`:

### In `PyADM1ODE_estimate` (already active)

```yaml
plugins:
  - mkdocstrings:
      handlers:
        python:
          paths: [.]
          import:
            - https://dgaida.github.io/PyADM1ODE/latest/objects.inv
            - https://dgaida.github.io/PyADM1ODE_calibration/latest/objects.inv
```

### In `PyADM1ODE_calibration` (to add)

```yaml
plugins:
  - mkdocstrings:
      handlers:
        python:
          paths: [.]
          import:
            - https://dgaida.github.io/PyADM1ODE/latest/objects.inv
            - https://dgaida.github.io/PyADM1ODE_estimate/latest/objects.inv
```

### In `PyADM1ODE` (to add)

```yaml
plugins:
  - mkdocstrings:
      handlers:
        python:
          paths: [.]
          import:
            - https://dgaida.github.io/PyADM1ODE_calibration/latest/objects.inv
            - https://dgaida.github.io/PyADM1ODE_estimate/latest/objects.inv
```

**Effect**: writing `[pyadm1.BiogasPlant][]` in Markdown will automatically
link to the right page in the base doc — even from the extension repos.
Dead URLs (e.g. when a repo is not yet published) only emit a warning, not
a build failure.

## Tactic 3 — Ecosystem navigation in every repo

In each repo's `mkdocs.yml`, add a last top-level entry `Ecosystem` with
an overview page and external links to the respective other two repos.

### `PyADM1ODE_estimate` (already active)

```yaml
nav:
  # ...
  - Ecosystem:
      - Overview: ecosystem.md
      - PyADM1ODE (base): https://dgaida.github.io/PyADM1ODE/latest/
      - PyADM1ODE_calibration: https://dgaida.github.io/PyADM1ODE_calibration/latest/
```

### `PyADM1ODE_calibration` (to add)

```yaml
nav:
  # ...
  - Ecosystem:
      - Overview: ecosystem.md
      - PyADM1ODE (base): https://dgaida.github.io/PyADM1ODE/latest/
      - PyADM1ODE_estimate: https://dgaida.github.io/PyADM1ODE_estimate/latest/
```

### `PyADM1ODE` (to add — as the "hub")

```yaml
nav:
  # ...
  - Ecosystem:
      - Overview: ecosystem.md
      - PyADM1ODE_calibration: https://dgaida.github.io/PyADM1ODE_calibration/latest/
      - PyADM1ODE_estimate: https://dgaida.github.io/PyADM1ODE_estimate/latest/
```

Plus a `docs/{lang}/ecosystem.md` page with the Mermaid diagram of the data
flow and a short introduction to the three components. Template: see
[ecosystem.md](../ecosystem.md) in this repo.

## Tactic 4 — Footer branding

A short footer line makes each repo visibly part of the ecosystem. In
`mkdocs.yml`:

### `PyADM1ODE_estimate` (already active)

```yaml
copyright: >
  Copyright &copy; 2026 Daniel Gaida — part of the
  <a href="https://dgaida.github.io/PyADM1ODE/latest/">PyADM1ODE ecosystem</a>
```

### `PyADM1ODE_calibration` and `PyADM1ODE` (to add)

Same scheme, with the link pointing to one of the other repos if you prefer.

## Tactic 5 — Keep `mike` versions in sync

When a release spans all three repos, deploy the same version number in all
three workflows. In `.github/workflows/docs.yml` the relevant step is:

```yaml
- name: Deploy Documentation (mike)
  if: github.event_name == 'push' && (github.ref == 'refs/heads/main' || startsWith(github.ref, 'refs/tags/v'))
  run: |
    git config --global user.name "github-actions[bot]"
    git config --global user.email "github-actions[bot]@users.noreply.github.com"
    git fetch origin gh-pages --depth=1 || true
    mike delete latest || true
    if [[ $GITHUB_REF == refs/tags/* ]]; then
      VERSION=${GITHUB_REF#refs/tags/}
      mike deploy --push --update-aliases $VERSION latest
    else
      mike deploy --push --update-aliases dev latest
    fi
    mike set-default --push latest
```

Identical in all three repos. Set tags via `git tag v0.2.0 && git push origin v0.2.0`
in all three repos simultaneously, and all three docs deploy to `/v0.2.0/`
and become reachable in parallel via the version selector.

## Checklist for a new repo in the ecosystem

If a fourth repo joins (e.g. `PyADM1ODE_control`):

- [ ] `mkdocs.yml` with the theme palette above
- [ ] i18n plugin (`docs/de/` + `docs/en/`)
- [ ] `mike` plugin with `provider: mike`
- [ ] `objects.inv` imports of the other three repos in the mkdocstrings handler
- [ ] Ecosystem top-level entry in `nav`
- [ ] `docs/{lang}/ecosystem.md` page (template: this repo)
- [ ] Footer copyright with "part of the PyADM1ODE ecosystem"
- [ ] `.github/workflows/docs.yml` analogous to the one used here
- [ ] In the other three repos: add the new repo to the `objects.inv` imports
  and to the ecosystem nav
