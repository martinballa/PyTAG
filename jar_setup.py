# requires the gdown module to be installed
import gdown, zipfile, argparse, os

argParser = argparse.ArgumentParser()
argParser.add_argument("-file-id", "--file-id", default="1uPNoZkdI4rJiFyNyXFVun_VcAlN3QIVQ",  help="your name")
args = argParser.parse_args()

filename = "pytag/jars/ModernBoardGame.zip"
file_id = args.file_id #"1uPNoZkdI4rJiFyNyXFVun_VcAlN3QIVQ"
gdown.download(
    f"https://drive.google.com/uc?export=download&confirm=pbef&id={file_id}",
    filename
)

# unzip the files into
jar_path = "pytag/"
with zipfile.ZipFile(filename, 'r') as zip_file:
    zip_file.extractall(jar_path)
os.remove(filename)