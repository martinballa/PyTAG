# PyTAG: a Reinforcement Learning interface for the [Tabletop Games Framework](http://www.tabletopgames.ai/)

[![license](https://img.shields.io/github/license/martinballa/PyTAG)](LICENSE)
![top-language](https://img.shields.io/github/languages/top/martinballa/PyTAG)
![code-size](https://img.shields.io/github/languages/code-size/martinballa/PyTAG)
[![twitter](https://img.shields.io/twitter/follow/gameai_qmul?style=social)](https://twitter.com/intent/follow?screen_name=gameai_qmul)
[![](https://img.shields.io/github/stars/martinballa/PyTAG.svg?label=Stars&style=social)](https://github.com/GAIGResearch/TabletopGames)

PyTAG allows interaction with the TAG framework from Python. This repository contains all the python code required to
run Reinforcement Learning agents.
The aim of PyTAG is to provide a Reinforcement Learning API for the TAG framework, but it is not limited to RL as using
the python-java bridge all public functions and variables are accessible from python.
If you want to learn more about TAG, please visit the [website](http://tabletopgames.ai).

You may try [this](https://colab.research.google.com/drive/1WMVu9bFkxvwK7evD1sIkxcsrlhdRoY9d?usp=sharing) google colab
notebook to try out PyTAG before installing it on your own machine.

## Setting up

TAG requires Java with minimum version 21. We recommend installing pytag in a new virtual environment. To 
install
pytag you may follow the steps below.

- 1, Clone this repository.
- 2, Install PyTAG as a python package ```pip install -e .```
- 3, Run ```python jar_setup.py``` to download the latest `TAG.jar` or see "Getting the TAG jar file" below for manual options.
- 4, (optional) install pytag with the additional dependencies to run the baselines ```pip install -e .[examples]```
- 5, (optional) you may test your installation by running the examples in ```examples/``` for instance
  ```pt-action-masking.py```.

### Getting the TAG jar file
PyTAG requires a single `TAG.jar` file placed in the `pytag/jars/` folder. Running `jar_setup.py` will download it automatically (no extra dependencies required):
```bash
python jar_setup.py
```
Or download `TAG.jar` manually from the [TAG releases page](https://github.com/GAIGResearch/TabletopGames/releases) and place it in `pytag/jars/`.

To build `TAG.jar` from source, see the [TAG wiki](https://tabletopgames.ai/wiki/maven): run `mvn package` in the TAG repository and copy `target/TAG-pytag.jar` to `pytag/jars/TAG.jar` (the slim PyTAG-specific build - not `target/TAG.jar`, which bundles unrelated ML/analytics dependencies PyTAG doesn't need).

#### Using a custom jar without touching `pytag/jars/`

If you're iterating on a local TAG checkout (e.g. testing a game or agent change before it's released),
you don't need to copy your build into `pytag/jars/TAG.jar` each time. Point PyTAG at it directly with the
`PYTAG_JAR_PATH` environment variable:

```bash
export PYTAG_JAR_PATH=/path/to/TabletopGames/target/TAG-pytag.jar
python examples/pt-action-masking.py
```

This takes priority over `pytag/jars/TAG.jar` for both `PyTAG`/`MultiAgentPyTAG`/`SelfPlayPyTAG` and
`list_supported_games()`. Unset it (or leave it unset) to fall back to the downloaded jar.

## Supported games

The following games are currently supported (as registered Gymnasium environments):

| Game | Gym ID | Obs type | Players |
|------|--------|----------|---------|
| Diamant | `TAG/Diamant-v0` | vector | 2+ |
| TicTacToe | `TAG/TicTacToe-v0` | vector | 2 |
| LoveLetter | `TAG/LoveLetter-v0` | vector | 2+ |
| Stratego | `TAG/Stratego-v0` | vector | 2 |
| SushiGo | `TAG/SushiGo-v0` | JSON | 2–5 |
| SushiGo (multi-agent) | `TAG/SushiGo-MA-v0` | JSON | 2 |
| PowerGrid | `TAG/PowerGrid-v0` | vector | 3–6 |

## Getting started

The `examples/` folder provides scripts to get started with the framework.
`pt-action-masking.py` demonstrates manual action masking; `gym-action-masking.py` extends this to a Gymnasium environment; `gym-random.py` uses the built-in action sampler; `ma-random.py` shows how to control multiple Python agents simultaneously.

The PPO baseline scripts (`ppo.py`, `ppo-lstm.py`, `ppo-selfplay.py`) reproduce the experiments from the papers listed below. `ppo-eval.py` loads a trained model for evaluation. The self-play script (`ppo-selfplay.py`) trains agents via self-play using `TAGSelfPlayGYm` and a checkpoint pool for opponent selection.

## Citing Information

If you use PyTAG in your work, please cite the relevant papers below.

The IEEE Transactions on Games journal paper covers the full framework including multiagent and self-play environments:
```bibtex
@article{balla2024pytag,
  author    = {Balla, Martin and Long, George E. M. and Goodman, James and Gaina, Raluca D. and Perez-Liebana, Diego},
  journal   = {IEEE Transactions on Games},
  title     = {{PyTAG}: Tabletop Games for Multiagent Reinforcement Learning},
  year      = {2024},
  volume    = {16},
  number    = {4},
  pages     = {993--1002},
  doi       = {10.1109/TG.2024.3404133}
}
```

The original CoG 2023 paper introduced the single-agent interface:
```bibtex
@inproceedings{balla2023pytag,
  author    = {Balla, Martin and Long, George E. M. and Jeurissen, Dominik and Goodman, James and Gaina, Raluca D. and Perez-Liebana, Diego},
  title     = {{PyTAG}: Challenges and Opportunities for Reinforcement Learning in Tabletop Games},
  booktitle = {IEEE Conference on Games (CoG)},
  year      = {2023}
}
```

To cite the TAG framework itself:
```bibtex
@inproceedings{gaina2020tag,
  author    = {Raluca D. Gaina and Martin Balla and Alexander Dockhorn and Raul Montoliu and Diego Perez-Liebana},
  title     = {{TAG}: A Tabletop Games Framework},
  booktitle = {Experimental AI in Games (EXAG), AIIDE 2020 Workshop},
  year      = {2020}
}
```

## Contact and contribute

The main method to contribute to our repository directly with code, or to suggest new features, point out bugs or ask
questions about the project is
through [creating new Issues on this github repository](https://github.com/GAIGResearch/TabletopGames/issues)
or [creating new Pull Requests](https://github.com/GAIGResearch/TabletopGames/pulls). Alternatively, you may contact the
authors of the papers listed above.

You can also find out more about the [QMUL Game AI Group](http://gameai.eecs.qmul.ac.uk/).

## Acknowledgements

This work was partly funded by the EPSRC CDT in Intelligent Games and Game Intelligence (IGGI)  EP/L015846/1 and EPSRC
research grant EP/T008962/1.
