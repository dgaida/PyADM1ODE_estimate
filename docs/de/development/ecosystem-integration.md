# Ecosystem-Integration

Diese Seite beschreibt, wie ein Repo des PyADM1ODE-Ökosystems sich mit den
anderen verbindet. Sie ist gleichzeitig die Migration-Anleitung, um die zwei
Schwester-Repos (`PyADM1ODE`, `PyADM1ODE_calibration`) auf denselben Stand wie
`PyADM1ODE_estimate` zu bringen.

## Übersicht — die fünf Tactics

| # | Tactic | Aufwand |
|---|---|---|
| 1 | Gleiches Material-Theme + Palette | < 5 min |
| 2 | `objects.inv`-Cross-References | < 10 min |
| 3 | Konsistente Top-Navigation (Ecosystem-Section) | < 10 min |
| 4 | Footer mit Eco-Branding | 2 min |
| 5 | `mike`-Versionen synchron halten | laufend, organisatorisch |

Konkrete Snippets unten.

## Tactic 1 — Theme-Palette angleichen

Alle drei Repos sollten in `mkdocs.yml` dieselbe Material-Konfiguration haben:

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

Die `language`-Einstellung pro Repo bleibt unverändert (`de` als Default,
i18n-Plugin macht den Switcher).

## Tactic 2 — Cross-API-References via `objects.inv`

`mkdocstrings` kann auf Python-Objekte einer anderen Doku verweisen, sobald
deren `objects.inv` veröffentlicht ist. In `mkdocs.yml`:

### In `PyADM1ODE_estimate` (bereits aktiv)

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

### In `PyADM1ODE_calibration` (ergänzen)

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

### In `PyADM1ODE` (ergänzen)

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

**Effekt**: schreibt man im Markdown-Code `[pyadm1.BiogasPlant][]`, wird der
Link automatisch auf die richtige Seite der Basis-Doku gesetzt — auch aus den
Erweiterungs-Repos heraus. Tote URLs (z.B. wenn ein Repo noch nicht
veröffentlicht ist) führen nur zu einem Warning, kein Build-Failure.

## Tactic 3 — Ecosystem-Navigation in jedem Repo

In `mkdocs.yml` jedes Repos einen letzten Top-Level-Eintrag `Ecosystem` mit
einer Übersichts-Seite und externen Links auf die jeweils anderen zwei Repos.

### `PyADM1ODE_estimate` (bereits aktiv)

```yaml
nav:
  # ...
  - Ecosystem:
      - Übersicht: ecosystem.md
      - PyADM1ODE (Basis): https://dgaida.github.io/PyADM1ODE/latest/
      - PyADM1ODE_calibration: https://dgaida.github.io/PyADM1ODE_calibration/latest/
```

### `PyADM1ODE_calibration` (ergänzen)

```yaml
nav:
  # ...
  - Ecosystem:
      - Übersicht: ecosystem.md
      - PyADM1ODE (Basis): https://dgaida.github.io/PyADM1ODE/latest/
      - PyADM1ODE_estimate: https://dgaida.github.io/PyADM1ODE_estimate/latest/
```

### `PyADM1ODE` (ergänzen — als „Hub")

```yaml
nav:
  # ...
  - Ecosystem:
      - Übersicht: ecosystem.md
      - PyADM1ODE_calibration: https://dgaida.github.io/PyADM1ODE_calibration/latest/
      - PyADM1ODE_estimate: https://dgaida.github.io/PyADM1ODE_estimate/latest/
```

Plus eine `docs/{lang}/ecosystem.md`-Seite mit dem Mermaid-Diagramm des
Datenflusses und kurzer Vorstellung der drei Komponenten. Vorlage siehe
[ecosystem.md](../ecosystem.md) in diesem Repo.

## Tactic 4 — Footer-Branding

Eine kurze Zeile im Footer macht in jedem Repo sichtbar, dass es Teil des
Ökosystems ist. In `mkdocs.yml`:

### `PyADM1ODE_estimate` (bereits aktiv)

```yaml
copyright: >
  Copyright &copy; 2026 Daniel Gaida — part of the
  <a href="https://dgaida.github.io/PyADM1ODE/latest/">PyADM1ODE ecosystem</a>
```

### `PyADM1ODE_calibration` und `PyADM1ODE` (ergänzen)

Gleiches Schema, ggf. Link auf eines der anderen Repos.

## Tactic 5 — `mike`-Versionen synchron halten

Wenn ein Release alle drei Repos zusammenführt, in allen drei Workflows
dieselbe Versionsnummer deployen. Im `.github/workflows/docs.yml` ist der
relevante Schritt:

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

Identisch in allen drei Repos. Tags via `git tag v0.2.0 && git push origin v0.2.0`
gleichzeitig in allen drei Repos setzen, dann werden alle drei Docs auf
`/v0.2.0/` deployt und sind über den Version-Selector parallel erreichbar.

## Checkliste für ein neues Repo im Ökosystem

Wenn ein viertes Repo dazu kommt (z.B. `PyADM1ODE_control`):

- [ ] `mkdocs.yml` mit obiger Theme-Palette
- [ ] i18n-Plugin (`docs/de/` + `docs/en/`)
- [ ] `mike`-Plugin mit `provider: mike`
- [ ] `objects.inv`-Imports der anderen drei Repos im mkdocstrings-Handler
- [ ] Ecosystem-Top-Level-Eintrag in `nav`
- [ ] `docs/{lang}/ecosystem.md`-Seite (Vorlage: dieses Repo)
- [ ] Footer-Copyright mit „part of the PyADM1ODE ecosystem"
- [ ] `.github/workflows/docs.yml` analog dem hier verwendeten
- [ ] In den anderen drei Repos: das neue Repo zu den `objects.inv`-Imports
  und zur Ecosystem-Nav hinzufügen
