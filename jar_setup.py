# requires the gdown module to be installed
import gdown, argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("-file-id", "--file-id", default="1wIM2xPE5tqvVzO931t3xcVYWk7VCr6i8", help="Google Drive file ID for TAG.jar")
args = argParser.parse_args()

filename = "pytag/jars/TAG.jar"
file_id = args.file_id
gdown.download(
    f"https://drive.google.com/uc?export=download&confirm=pbef&id={file_id}",
    filename
)
