from os.path import join, abspath, dirname, pardir
BASE_DIR = abspath(join(dirname(__file__), pardir))
output_dir = '/data/Deep_fingerprint/processed_RF_data/'
split_mark = '\t'
OPEN_WORLD = True
MONITORED_SITE_NUM = 100
model_path = 'pretrained/'

num_classes = 100
num_classes_ow = 101
# Length of TAM
max_matrix_len = 1800
# Maximum Load Time
maximum_load_time = 80

max_trace_length = 5000
