# PyTAG: a Reinforcement Learning interface for the [Tabletop Games Framework](http://www.tabletopgames.ai/)

[![license](https://img.shields.io/github/license/martinballa/PyTAG)](LICENSE)
![top-language](https://img.shields.io/github/languages/top/martinballa/PyTAG)
![code-size](https://img.shields.io/github/languages/code-size/martinballa/PyTAG)
[![twitter](https://img.shields.io/twitter/follow/gameai_qmul?style=social)](https://twitter.com/intent/follow?screen_name=gameai_qmul)
[![](https://img.shields.io/github/stars/martinballa/PyTAG.svg?label=Stars&style=social)](https://github.com/GAIGResearch/TabletopGames)


PyTAG allows interaction with the TAG framework from Python. This repository contains all the python code required to run Reinforcement Learning agents.
The aim of PyTAG is to provide a Reinforcement Learning API for the TAG framework, but it is not limited to RL as using the python-java bridge all public functions and variables are accessible from python.
If you want to learn more about TAG, please visit the [website](http://tabletopgames.ai).

You may try [this](https://colab.research.google.com/drive/1WMVu9bFkxvwK7evD1sIkxcsrlhdRoY9d?usp=sharing) google colab notebook to try out PyTAG before installing it on your own machine.

## Setting up
The project requires Java with minimum version 8. To install pytag you may follow the steps below.
- 1, Clone this repository.
- 2, Install PyTAG as a python package ```python pytag/setup.py develop```
- 3, Run ```jar_setup.py``` to download the latest jar file for TAG (requires installing the ```gdown``` python module) or see the section on "Getting the TAG jar files" below for more options.
- 4, (optional) install pytag with the additional dependencies to run the baselines ```python pytag/setup.py develop easy_install "pytag[examples]"```
- 5, (optional) you may test your installation by running the examples in ```examples/``` for instance ```pt-action-masking.py```.

### Getting the TAG jar files
Pytag is looking for the TAG jar files in the ```pytag/jars/``` folder. To get the latest jar files you may run ```jar_setup.py``` which will download the latest jar files and unpack them at the correct location.
Or alternatively you may manually download it from [Google drive](https://drive.google.com/file/d/1uPNoZkdI4rJiFyNyXFVun_VcAlN3QIVQ/view?usp=drive_link)  and place the jar files in the ```pytag/jars/``` folder.

In case that you want to make changes to the JAVA framework (i.e.: implementing the RL interfaces for a new game) you need to create new jar files from TAG and place them in the ```pytag/jars/``` folder.

### Installing PyTAG
Note that in the above options we used ```python pytag/setup.py develop``` to install PyTAG. This will install PyTAG as a python package in development mode. This means that you can make changes to the code and it will be reflected in your python environment without having to reinstall the package. If you want to install PyTAG as a regular python package you may use ```python pytag/setup.py install``` instead.

## Getting started

The examples folder provides a few python scripts that may serve as a starting point for using the framework. 
```pt-action-masking.py``` demonstrates how the action masking may be used to sample random valid actions manually. ```gym-action-masking.py``` extends this to using the action masking in a gym environment. ```gym-random.py``` shows how the built-in action sampler may be used.
```ma-random.py``` demonstrates how multiple python agents may be controlled.
The remaining scripts are used to run the PPO baselines from the IEEE CoG 23' paper. ```ppo-eval.py``` allows you to load a trained PPO model for evaluation.

## Citing Information

To cite PyTAG in your work, please cite this paper:
```
@article{balla2023pytag,
  title={PyTAG: Challenges and Opportunities for Reinforcement Learning in Tabletop Games},
  author={Balla, Martin and Long, George EM and Jeurissen, Dominik and Goodman, James and Gaina, Raluca D and Perez-Liebana, Diego},
  year= {2023},
  booktitle= {{IEEE Conference on Games (CoG), 2023}},
}
```

To cite TAG in your work, please cite this paper:
```
@inproceedings{gaina2020tag,
         author= {Raluca D. Gaina and Martin Balla and Alexander Dockhorn and Raul Montoliu and Diego Perez-Liebana},
         title= {{TAG: A Tabletop Games Framework}},
         year= {2020},
         booktitle= {{Experimental AI in Games (EXAG), AIIDE 2020 Workshop}},
         abstract= {Tabletop games come in a variety of forms, including board games, card games, and dice games. In recent years, their complexity has considerably increased, with many components, rules that change dynamically through the game, diverse player roles, and a series of control parameters that influence a game's balance. As such, they also encompass novel and intricate challenges for Artificial Intelligence methods, yet research largely focuses on classical board games such as chess and Go. We introduce in this work the Tabletop Games (TAG) framework, which promotes research into general AI in modern tabletop games, facilitating the implementation of new games and AI players, while providing analytics to capture the complexities of the challenges proposed. We include preliminary results with sample AI players, showing some moderate success, with plenty of room for improvement, and discuss further developments and new research directions.},
    }
```

## Contact and contribute
The main method to contribute to our repository directly with code, or to suggest new features, point out bugs or ask questions about the project is through [creating new Issues on this github repository](https://github.com/GAIGResearch/TabletopGames/issues) or [creating new Pull Requests](https://github.com/GAIGResearch/TabletopGames/pulls). Alternatively, you may contact the authors of the papers listed above. 

You can also find out more about the [QMUL Game AI Group](http://gameai.eecs.qmul.ac.uk/).

## Acknowledgements

This work was partly funded by the EPSRC CDT in Intelligent Games and Game Intelligence (IGGI)  EP/L015846/1 and EPSRC research grant EP/T008962/1.
