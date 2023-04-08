# Iteration limits: 50, 75, 100, 300, 1000
python scripts/generate.py cfg/primitives/model_v2.yaml -d cfg/poke_experiments/mug/mug_v6.yaml -m test -o out/experiments/poke_experiments/model_v2/mug_v6/test_params/iter_lims/iter_50/ --gen_args "{"embed_weight": 0.0, "def_loss": 0.0, "iter_limit": 50, "conv_eps": 0.0}"
python scripts/generate.py cfg/primitives/model_v2.yaml -d cfg/poke_experiments/mug/mug_v6.yaml -m test -o out/experiments/poke_experiments/model_v2/mug_v6/test_params/iter_lims/iter_75/ --gen_args "{"embed_weight": 0.0, "def_loss": 0.0, "iter_limit": 75, "conv_eps": 0.0}"
python scripts/generate.py cfg/primitives/model_v2.yaml -d cfg/poke_experiments/mug/mug_v6.yaml -m test -o out/experiments/poke_experiments/model_v2/mug_v6/test_params/iter_lims/iter_100/ --gen_args "{"embed_weight": 0.0, "def_loss": 0.0, "iter_limit": 100, "conv_eps": 0.0}"
python scripts/generate.py cfg/primitives/model_v2.yaml -d cfg/poke_experiments/mug/mug_v6.yaml -m test -o out/experiments/poke_experiments/model_v2/mug_v6/test_params/iter_lims/iter_300/ --gen_args "{"embed_weight": 0.0, "def_loss": 0.0, "iter_limit": 300, "conv_eps": 0.0}"
python scripts/generate.py cfg/primitives/model_v2.yaml -d cfg/poke_experiments/mug/mug_v6.yaml -m test -o out/experiments/poke_experiments/model_v2/mug_v6/test_params/iter_lims/iter_1000/ --gen_args "{"embed_weight": 0.0, "def_loss": 0.0, "iter_limit": 1000, "conv_eps": 0.0}"

# Convergence epsilon: 1e-1, 1e-2, 1e-3, 1e-4, 1e-5
python scripts/generate.py cfg/primitives/model_v2.yaml -d cfg/poke_experiments/mug/mug_v6.yaml -m test -o out/experiments/poke_experiments/model_v2/mug_v6/test_params/conv_eps/eps_1e1/ --gen_args "{"embed_weight": 0.0, "def_loss": 0.0, "iter_limit": 1000, "conv_eps": 1e-1}"
python scripts/generate.py cfg/primitives/model_v2.yaml -d cfg/poke_experiments/mug/mug_v6.yaml -m test -o out/experiments/poke_experiments/model_v2/mug_v6/test_params/conv_eps/eps_1e2/ --gen_args "{"embed_weight": 0.0, "def_loss": 0.0, "iter_limit": 1000, "conv_eps": 1e-2}"
python scripts/generate.py cfg/primitives/model_v2.yaml -d cfg/poke_experiments/mug/mug_v6.yaml -m test -o out/experiments/poke_experiments/model_v2/mug_v6/test_params/conv_eps/eps_1e3/ --gen_args "{"embed_weight": 0.0, "def_loss": 0.0, "iter_limit": 1000, "conv_eps": 1e-3}"
python scripts/generate.py cfg/primitives/model_v2.yaml -d cfg/poke_experiments/mug/mug_v6.yaml -m test -o out/experiments/poke_experiments/model_v2/mug_v6/test_params/conv_eps/eps_1e4/ --gen_args "{"embed_weight": 0.0, "def_loss": 0.0, "iter_limit": 1000, "conv_eps": 1e-4}"
python scripts/generate.py cfg/primitives/model_v2.yaml -d cfg/poke_experiments/mug/mug_v6.yaml -m test -o out/experiments/poke_experiments/model_v2/mug_v6/test_params/conv_eps/eps_1e5/ --gen_args "{"embed_weight": 0.0, "def_loss": 0.0, "iter_limit": 1000, "conv_eps": 1e-5}"

# Embedding weight: 1e-4, 1e-3, 1e-2, 1e-1, 1.0
python scripts/generate.py cfg/primitives/model_v2.yaml -d cfg/poke_experiments/mug/mug_v6.yaml -m test -o out/experiments/poke_experiments/model_v2/mug_v6/test_params/embed_weight/w_1e4/ --gen_args "{"embed_weight": 0.0001, "def_loss": 0.0, "iter_limit": 1000, "conv_eps": 0.0}"
python scripts/generate.py cfg/primitives/model_v2.yaml -d cfg/poke_experiments/mug/mug_v6.yaml -m test -o out/experiments/poke_experiments/model_v2/mug_v6/test_params/embed_weight/w_1e3/ --gen_args "{"embed_weight": 0.001, "def_loss": 0.0, "iter_limit": 1000, "conv_eps": 0.0}"
python scripts/generate.py cfg/primitives/model_v2.yaml -d cfg/poke_experiments/mug/mug_v6.yaml -m test -o out/experiments/poke_experiments/model_v2/mug_v6/test_params/embed_weight/w_1e2/ --gen_args "{"embed_weight": 0.01, "def_loss": 0.0, "iter_limit": 1000, "conv_eps": 0.0}"
python scripts/generate.py cfg/primitives/model_v2.yaml -d cfg/poke_experiments/mug/mug_v6.yaml -m test -o out/experiments/poke_experiments/model_v2/mug_v6/test_params/embed_weight/w_1e1/ --gen_args "{"embed_weight": 0.1, "def_loss": 0.0, "iter_limit": 1000, "conv_eps": 0.0}"
python scripts/generate.py cfg/primitives/model_v2.yaml -d cfg/poke_experiments/mug/mug_v6.yaml -m test -o out/experiments/poke_experiments/model_v2/mug_v6/test_params/embed_weight/w_1/ --gen_args "{"embed_weight": 1.0, "def_loss": 0.0, "iter_limit": 1000, "conv_eps": 0.0}"