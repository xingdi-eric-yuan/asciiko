import codecs
import bitstring
from PIL import Image, ImageEnhance
import os
from collections import Counter, OrderedDict
from tqdm import tqdm


# Running this script will parse ETL2 data and generate hand written images for various characters.
# Each character image will be saved under ./ETL2_out/<char_name>/ directory.
# Note that this script assumes that ETL2 data is already downloaded to asciiko/char_classifier/etl2_reader/
# from this url http://etlcdb.db.aist.go.jp/?page_id=56
# Most of the parsing script is from this url: http://etlcdb.db.aist.go.jp/?page_id=1721


# Meta data containig file path and number of records (Each record contains one image for a character)
# [file to path, number of records]
ETL2C_META = (('./ETL2/ETL2_1', 9056),
              ('./ETL2/ETL2_2', 10480),
              ('./ETL2/ETL2_3', 11360),
              ('./ETL2/ETL2_4', 10480),
              ('./ETL2/ETL2_5', 11420))


out_dir = "ETL2_out"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


t56s = '0123456789[#@:>? ABCDEFGHI&.](<  JKLMNOPQR-$*);\'|/STUVWXYZ ,%="!'
def T56(c):
    return t56s[c]


with codecs.open('co59-utf8.txt', 'r', 'utf-8') as co59f:
    co59t = co59f.read()
co59l = co59t.split()
CO59 = {}


for c in co59l:
    ch = c.split(':')
    co = ch[1].split(',')
    CO59[(int(co[0]), int(co[1]))] = ch[0]


char_freq = []

for meta in ETL2C_META:

    filename = meta[0]

    f = bitstring.ConstBitStream(filename=filename)
    print("Reading {}".format(filename))
    for skip in tqdm(range(meta[1])):
        f.pos = skip * 6 * 3660
        r = f.readlist('int:36,uint:6,pad:30,6*uint:6,6*uint:6,pad:24,2*uint:6,pad:180,bytes:2700')
        #print(r[0], T56(r[1]), "".join(map(T56, r[2:8])), "".join(map(T56, r[8:14])), CO59[tuple(r[14:16])])

        char_name = CO59[tuple(r[14:16])]

        char_freq.append(char_name)

        output_char_dir = os.path.join(out_dir, char_name)
        if not os.path.exists(output_char_dir):
            os.makedirs(output_char_dir)

        iF = Image.frombytes('F', (60, 60), r[16], 'bit', 6)
        iP = iF.convert('RGBA')
        fn = '{:d}.png'.format(r[0])
        # iP.save(fn, 'PNG', bits=6)

        enhancer = ImageEnhance.Brightness(iP)
        iE = enhancer.enhance(4)
        iE.save(os.path.join(output_char_dir, fn), 'PNG')


counter = Counter(char_freq)
od = OrderedDict(sorted(counter.items()))
stat_file_path = os.path.join(out_dir, "etl2_data_stat.txt")
with open(stat_file_path, 'w') as f:
    for char, cnt in od.items():
        f.write("{}: {}\n".format(char, cnt))
print("Saved stats to {}".format(stat_file_path))
