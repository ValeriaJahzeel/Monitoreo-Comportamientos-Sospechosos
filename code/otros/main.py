import code.otros.getKeypoints as gk
import code.otros.getFrame as gf
import code.preprocesamientoDatos.preprocesamientoVideo as pv

video = "./dataset/sospechoso/"
frames = './informacion//frames/sospechoso/'
csv_file = './informacion/csv/sospechoso/'
trazos = './informacion/trazos/sospechoso/'

# gk.framesVideos(video,frames, csv_file, trazos)
pv.main(video,frames, csv_file, trazos)
#gf.ObtenerFrames(video,frames, trazos)

