# PyTAG: a Reinforcement Learning interface for the [Tabletop Games Framework](http://www.tabletopgames.ai/)

[![license](https://img.shields.io/github/license/martinballa/PyTAG)](LICENSE)
![top-language](https://img.shields.io/github/languages/top/martinballa/PyTAG)
![code-size](https://img.shields.io/github/languages/code-size/martinballa/PyTAG)
[![twitter](https://img.shields.io/twitter/follow/gameai_qmul?style=social)](https://twitter.com/intent/follow?screen_name=gameai_qmul)
[![](https://img.shields.io/github/stars/martinballa/PyTAG.svg?label=Stars&style=social)](https://github.com/GAIGResearch/TabletopGames)


PyTAG allows interaction with the TAG framework from Python. This repository contains all the python code required to run Reinforcement Learning agents.
The aim of PyTAG is to provide a Reinforcement Learning API for the TAG framework, but it is not limited to RL as using the python-java bridge all public functions and variables are accessible from python.
If you want to learn more about TAG, please visit the [website](http://tabletopgames.ai).

## Setting up
The project requires Java with minimum version 8. Currently, to run PyTAG you need to set up a few things manually.
1, Clone this repository.
2, Download the latest jar file for [TAG](https://drive.google.com/file/d/16VVSEKUXj4lx-iniAprkSdttPzIYSxMN/view?usp=drive_link) 
3, Place the jar file in the ```pytag/jars/``` folder.
4, Install PyTAG as a python package ```pip install -e pytag/ ```
5, (optional) you may test your installation by running the examples in ```examples/```

In the future we are hoping to automate the installation process and make PyTAG more accessible. 

## Getting started

The examples folder provides a few python scripts that may serve as a starting point for using the framework. PPO and PPO_LSTM were used as baselines in the CoG 23' paper. 

## Modifying TAG
In case that you want to make changes to the JAVA framework (i.e.: implementing the RL interfaces for a new game) you may replace the jar file in ```pytag/jars/``` with the updated one.

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
