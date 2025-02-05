import getKeypoints as gk
import getFrame as gf

video = "./dataset/sospechoso/"
frames = './informacion//frames/sospechoso/'
csv_file = './informacion/csv/sospechoso/'
trazos = './informacion/trazos/sospechoso/'

gk.framesVideos(video,frames, csv_file, trazos)
#gf.ObtenerFrames(video,frames, trazos)

