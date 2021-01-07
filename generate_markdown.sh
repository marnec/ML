mkdir -p docs/pages/data
jupyter nbconvert ML-*.ipynb --to markdown --output-dir='./docs/pages' --TagRemovePreprocessor.remove_cell_tags='{"remove_cell"}' --TagRemovePreprocessor.remove_input_tags='{"remove_input"}' --TagRemovePreprocessor.remove_all_outputs_tags='{"remove_output"}'
rsync -rtv data/ docs/pages/data
