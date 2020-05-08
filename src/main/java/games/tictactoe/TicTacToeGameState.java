package games.tictactoe;

import actions.IAction;
import actions.SetGridValueAction;
import components.Grid;
import core.AbstractGameState;
import gamestates.PlayerResult;
import gamestates.GridGameState;
import observations.GridObservation;
import observations.Observation;
import players.AbstractPlayer;

import java.util.*;
import java.util.List;


public class TicTacToeGameState extends AbstractGameState implements GridGameState<Character> {

    private Grid<Character> grid = new Grid<>(3, 3, ' ');

    //HashMap<AbstractPlayer, Character> playerSymbols = new HashMap<>();

    public TicTacToeGameState(TicTacToeGameParameters gameParameters){
        super(gameParameters);
    }

    @Override
    public Observation getObservation(AbstractPlayer player) {
        return new GridObservation<>(grid.getGridValues());
    }

    @Override
    public List<IAction> getActions(AbstractPlayer player) {
        ArrayList<IAction> actions = new ArrayList<>();

        for (int x = 0; x < grid.getWidth(); x++){
            for (int y = 0; y < grid.getHeight(); y++) {
                if (grid.getElement(x, y) == ' ')
                    actions.add(new SetGridValueAction<>(grid, x, y, player.playerID == 0 ? 'x' : 'o'));
            }
        }
        return actions;
    }

    @Override
    public void endGame() {
        terminalState = true;
        Arrays.fill(playerResults, PlayerResult.Draw);
    }

    @Override
    public Grid<Character> getGrid() {
        return grid;
    }

    public void registerWinner(char winnerSymbol){
        terminalState = true;
        if (winnerSymbol == 'o'){
            playerResults[1] = PlayerResult.Winner;
            playerResults[0] = PlayerResult.Loser;
        } else {
            playerResults[0] = PlayerResult.Winner;
            playerResults[1] = PlayerResult.Loser;
        }
    }
}