# импорт библиотек
import time
from pathlib import Path

from matching import Matching
from utils import save_result_csv

from config import crop_name, layout_name, save_path, clahe, del_bad_pixels


# основная функция
def run_processing(crop_name: Path, layout_name: Path, save_path: Path) -> int:
    # запускаем таймер
    start_time = time.time()

    # проверяем все пути
    if not Path(crop_name).exists():
        raise FileNotFoundError(f"No image at path {crop_name}.")

    if not Path(layout_name).exists():
        raise FileNotFoundError(f"No image at path {layout_name}.")

    #if not Path(save_path.parents[0]).exists():
    #    raise FileNotFoundError(f"No image at path {save_path}.")

    matching = Matching(layout_name, crop_name, save_path, clahe=clahe, del_bad_pixels=del_bad_pixels)
    dst = matching.run_processing()
    save_result_csv(layout_name, crop_name, dst, save_path, start_time, time.time())


if __name__ == '__main__':
    # arg_list = sys.argv[:]
    #
    # parser = argparse.ArgumentParser(description='creator_CD')
    # parser.add_argument('-v', '-version', action='version', version='%(prog)s 0.1')
    #
    # parser.add_argument('-crop_name', action='store', help='full path to crop tif',
    #                     required=False)
    #
    # parser.add_argument('-layout_name', action='store', help='full path to layout path',
    #                     required=False)
    #
    # parser.add_argument('-save_path', action='store', help='full path to save GeoTiff',
    #                     required=False)
    #
    # args = parser.parse_args()
    #
    # ret = run_processing(crop_name=args.crop_name,
    #                 layout_name=args.layout_name,
    #                 save_path=args.save_path)

    ret = run_processing(
        crop_name=crop_name,
        layout_name=layout_name,
        save_path=save_path
        )
