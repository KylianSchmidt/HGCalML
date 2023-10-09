# !/bin/bash
exit(0)

python3 ./normalize_inputs.py ./nntr_data/normal_detector/Raw/Training.root
python3 ./normalize_inputs.py ./nntr_data/normal_detector/Raw/Testing.root
python3 ./normalize_inputs.py ./nntr_data/idealized_detector/Raw/Training.root
python3 ./normalize_inputs.py ./nntr_data/idealized_detector/Raw/Testing.root
