mkdir -p docs/pages/data
jupyter nbconvert ML-*.ipynb --to markdown --output-dir='./docs/pages' --TagRemovePreprocessor.remove_cell_tags='{"remove_cell"}' --TagRemovePreprocessor.remove_input_tags='{"remove_input"}' --TagRemovePreprocessor.remove_all_outputs_tags='{"remove_output"}'
sed -i  '1{/^[[:space:]]*$/d}' docs/pages/*ML-*.md
sed -i 's/<\/input>//g' docs/pages/*ML-*.md
python3 figcaption.py docs/pages/*ML-*.md
python3 addCodeHeader.py docs/pages/*ML-*.md
rsync -rtv data/ docs/pages/data
git add --all
