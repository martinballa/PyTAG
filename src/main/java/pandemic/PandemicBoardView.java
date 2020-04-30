package pandemic;

import components.*;
import content.*;
import core.GameState;
import utilities.Hash;
import utilities.ImageIO;
import utilities.Utils;
import utilities.Vector2D;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.Point2D;
import java.util.ArrayList;

import static pandemic.PandemicCardView.drawCard;

public class PandemicBoardView extends JComponent {
    //TODO: images for tokens?
    
    private Image background;
    private IBoard board;
    private int width;
    private int height;
    PandemicGameState gameState;
    int nodeSize = 20;
    int researchStationSize = 10;
    int playerPawnSize = 10;
    int diseaseCubeSize = 10;
    int diseaseCubeDistance = 2;
    int counterWidth = 20, counterHeight = 20;

    Point2D[] infectionPositions = new Point2D[]{
            new Point2D.Double(755, 180),
            new Point2D.Double(795, 180),
            new Point2D.Double(835, 180),
            new Point2D.Double(875, 180),
            new Point2D.Double(915, 180),
            new Point2D.Double(955, 180),
            new Point2D.Double(995, 180)
    };
    Point2D[] outbreakPositions = new Point2D[]{
            new Point2D.Double(75, 450),
            new Point2D.Double(120, 495),
            new Point2D.Double(75, 530),
            new Point2D.Double(120, 565),
            new Point2D.Double(75, 600),
            new Point2D.Double(120, 630),
            new Point2D.Double(75, 665),
            new Point2D.Double(120, 700),
            new Point2D.Double(75, 730)
    };

    Point2D infectionDiscardPosition = new Point2D.Double(915, 50);
    Point2D playerDiscardPosition = new Point2D.Double(880, 625);
    Point2D[] diseaseMarkerPositions = new Point2D[]{
            new Point2D.Double(395, 775),
            new Point2D.Double(450, 775),
            new Point2D.Double(510, 775),
            new Point2D.Double(560, 775)
    };

    public PandemicBoardView(GameState gs, String backgroundPath) {
        gameState = (PandemicGameState) gs;
        this.board = ((PandemicGameState) gs).getData().findBoard("Cities");
        this.background = ImageIO.GetInstance().getImage(backgroundPath);
        width = background.getWidth(null);
        height = background.getHeight(null);
    }

    @Override
    protected void paintComponent(Graphics g) {
        drawBoard((Graphics2D) g);
    }

    private void drawBoard(Graphics2D g) {
        int fontSize = g.getFont().getSize();
        int nPlayers = gameState.getNPlayers();

        // Draw board background
        g.drawImage(background, 0, 0, null, null);

        // Draw nodes
        java.util.List<BoardNode> bList = board.getBoardNodes();
        for (BoardNode b : bList) {
            Vector2D pos = ((PropertyVector2D) ((BoardNode)b).getProperty(Hash.GetInstance().hash("coordinates"))).values;
//            g.setColor(Utils.stringToColor(((PropertyColor) b.getProperty(Hash.GetInstance().hash("color"))).valueStr));
//            g.fillOval(pos.getX() - nodeSize /2, pos.getY() - nodeSize /2, nodeSize, nodeSize);

            // Check if a research stations is here, draw just underneath the node
            PropertyBoolean isStation = (PropertyBoolean) ((BoardNode)b).getProperty(Hash.GetInstance().hash("researchStation"));
            if (isStation.value) {
                // Draw research station here
                g.setColor(Color.WHITE);
                g.fillRect(pos.getX() - researchStationSize /2, pos.getY() + nodeSize/2, researchStationSize, researchStationSize);
                g.setColor(Color.black);
                g.drawRect(pos.getX() - researchStationSize /2, pos.getY() + nodeSize/2, researchStationSize, researchStationSize);
                g.drawString("R", pos.getX() - researchStationSize /2 + 2, pos.getY() + nodeSize/2 + fontSize/2 + researchStationSize/2);
            }

            // Check if there are players here
            PropertyIntArrayList prop = (PropertyIntArrayList) ((BoardNode)b).getProperty(Constants.playersBNHash);
            ArrayList<Integer> players = prop.getValues();
            for (int p: players) {
                // This player is here, draw them just above the node
                // Find color of player
                Card playerCard = (Card) gameState.getAreas().get(p).getComponent(Hash.GetInstance().hash("playerCard"));
                PropertyColor color = (PropertyColor) playerCard.getProperty(Hash.GetInstance().hash("color"));
                g.setColor(Utils.stringToColor(color.valueStr));
                g.fillOval(pos.getX() + nPlayers * playerPawnSize / 2 - p * playerPawnSize - playerPawnSize /2, pos.getY() - nodeSize /2 - playerPawnSize /2, playerPawnSize, playerPawnSize);
                g.setColor(Color.black);
                g.drawOval(pos.getX() + nPlayers * playerPawnSize / 2 - p * playerPawnSize - playerPawnSize /2, pos.getY() - nodeSize /2 - playerPawnSize /2, playerPawnSize, playerPawnSize);
            }

            // Draw disease cubes on top of the node
            int[] array = ((PropertyIntArray) ((BoardNode)b).getProperty(Hash.GetInstance().hash("infection"))).getValues();
            int total = 0;
            for (int cube: array) {
                total += cube;
            }
            int idx = 0;
            int maxX = pos.getX() + (total + diseaseCubeDistance) * diseaseCubeSize / 4;
            for (int c = 0; c < array.length; c++) {
                int cube = array[c];
                Color cubeColor = Utils.stringToColor(Constants.colors[c]);
                for (int i = 0; i < cube; i++) {
                    g.setColor(cubeColor);
                    g.fillRect(maxX - idx * (diseaseCubeSize + diseaseCubeDistance) - diseaseCubeSize/2, pos.getY() - diseaseCubeSize /2, diseaseCubeSize, diseaseCubeSize);
                    g.setColor(Color.white);
                    if (cubeColor != null && cubeColor.equals(Color.yellow)) g.setColor(Color.black);  // Contrasting outline
                    g.drawRect(maxX - idx * (diseaseCubeSize + diseaseCubeDistance) - diseaseCubeSize/2, pos.getY() - diseaseCubeSize /2, diseaseCubeSize, diseaseCubeSize);
                    idx++;
                }

            }
        }

        // Draw infection rate marker
        Counter infectionRateCounter = (Counter) gameState.getAreas().get(-1).getComponent(Constants.infectionRateHash);
        Point2D pos = infectionPositions[infectionRateCounter.getValue()];
        g.drawImage(ImageIO.GetInstance().getImage("data/infectionRate.png"), (int)pos.getX(), (int)pos.getY(), null, null);

        // Draw outbreak marker
        Counter outbreakCounter = (Counter) gameState.getAreas().get(-1).getComponent(Constants.outbreaksHash);
        pos = outbreakPositions[outbreakCounter.getValue()];
        g.drawImage(ImageIO.GetInstance().getImage("data/outbreakMarker.png"), (int)pos.getX(), (int)pos.getY(), null, null);

        // Discard piles
        Deck playerDiscardDeck = (Deck) gameState.getAreas().get(-1).getComponent(Constants.playerDeckDiscardHash);
        if (playerDiscardDeck != null) {
            Card cP = (Card) playerDiscardDeck.draw();
            if (cP != null) {
                drawCard(g, 100, 50, cP, null, (int)playerDiscardPosition.getX(), (int)playerDiscardPosition.getY());
            }
        }
        Deck infectionDiscardDeck = (Deck) gameState.getAreas().get(-1).getComponent(Constants.infectionDiscardHash);
        if (infectionDiscardDeck != null) {
            Card cI = (Card) infectionDiscardDeck.draw();
            if (cI != null) {
                drawCard(g, 100, 50, cI, null, (int)infectionDiscardPosition.getX(), (int)infectionDiscardPosition.getY());
            }
        }

        // Disease markers
        Counter yC = (Counter) gameState.getAreas().get(-1).getComponent(Hash.GetInstance().hash("Disease yellow"));
        drawCounter(g, yC.getValue(), Color.yellow, 0);

        Counter rC = (Counter) gameState.getAreas().get(-1).getComponent(Hash.GetInstance().hash("Disease yellow"));
        drawCounter(g, rC.getValue(), Color.red, 1);

        Counter bC = (Counter) gameState.getAreas().get(-1).getComponent(Hash.GetInstance().hash("Disease yellow"));
        drawCounter(g, bC.getValue(), Color.blue, 2);

        Counter bkC = (Counter) gameState.getAreas().get(-1).getComponent(Hash.GetInstance().hash("Disease yellow"));
        drawCounter(g, bkC.getValue(), Color.black, 3);

    }

    private void drawCounter(Graphics2D g, int value, Color color, int idx) {
        if (value > 0) {
            Point2D pos = diseaseMarkerPositions[idx];
            g.setColor(color);
            g.fillOval((int)pos.getX(), (int)pos.getY(), counterWidth, counterHeight);
            if (value == 2) {
                g.setColor(Color.white);
                g.drawLine((int)pos.getX(), (int)pos.getY(), (int)pos.getX() + counterWidth, (int)pos.getY() + counterHeight);
            }
            g.setColor(Color.black);
        }
    }

    @Override
    public Dimension getPreferredSize() {
        return new Dimension(width, height);
    }
}
