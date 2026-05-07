# Authoring Examples

Examples live under `docs/examples/<section>/<name>.py` in jupytext
percent-format. Section folders contain a `_section.md` whose first heading
becomes the section title in the index.

## Quick start

```bash
mkdir -p docs/examples/my-section
echo '# My section\n\nIntro.\n' > docs/examples/my-section/_section.md
cat > docs/examples/my-section/hello.py <<'EOF'
# %% [markdown]
# # My example
#
# Short description.

# %%
print("hi")
EOF
just gallery
```

After running `just gallery`, browse `docs/examples/_built/` for the rendered
output. The first markdown heading in each example becomes the page title; the
next non-empty markdown line is used as the index-card summary.

## Tips

- Files starting with `_` are skipped by the discovery pass — handy for
  drafts.
- ``viewer.screenshot('out.png')`` is the supported way to capture napari
  views; CI runs every example under ``xvfb-run``.
- Keep examples self-contained: prefer synthetic data or fixtures from
  ``confusius.datasets`` over local files.
