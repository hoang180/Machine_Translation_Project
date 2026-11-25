"""
Script: data/download_data.py
Phụ trách: Phần 1 (Lê)
Chức năng: Tải tập dữ liệu chuẩn IWSLT'15 En-Vi từ GitHub Mirror (stefan-it/nmt-en-vi)
và giải nén trực tiếp ra thư mục data/raw/ gồm:
- train.en, train.vi (~133.000 cặp câu)
- tst2012.en, tst2012.vi (1.553 cặp câu - validation)
- tst2013.en, tst2013.vi (1.268 cặp câu - test)
"""

import os
import tarfile
import urllib.request
from tqdm import tqdm

DATA_DIR = "data/raw"

# Mirror đáng tin cậy trên GitHub (chuẩn 100% dữ liệu gốc IWSLT15 của Luong et al.)
ARCHIVES = [
    {
        "name": "dev-2012-en-vi.tgz",
        "url": "https://raw.githubusercontent.com/stefan-it/nmt-en-vi/master/data/dev-2012-en-vi.tgz",
        "expected_files": ["tst2012.en", "tst2012.vi"],
    },
    {
        "name": "test-2013-en-vi.tgz",
        "url": "https://raw.githubusercontent.com/stefan-it/nmt-en-vi/master/data/test-2013-en-vi.tgz",
        "expected_files": ["tst2013.en", "tst2013.vi"],
    },
    {
        "name": "train-en-vi.tgz",
        "url": "https://raw.githubusercontent.com/stefan-it/nmt-en-vi/master/data/train-en-vi.tgz",
        "expected_files": ["train.en", "train.vi"],
    },
]


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_and_extract():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("=" * 60)
    print(f"BẮT ĐẦU TẢI & GIẢI NÉN DỮ LIỆU IWSLT'15 EN-VI VÀO: {DATA_DIR}")
    print("=" * 60)

    for item in ARCHIVES:
        archive_name = item["name"]
        url = item["url"]
        expected_files = item["expected_files"]

        # Kiểm tra xem các file bên trong đã được giải nén chưa
        all_extracted = all(
            os.path.exists(os.path.join(DATA_DIR, f)) and os.path.getsize(os.path.join(DATA_DIR, f)) > 0
            for f in expected_files
        )

        if all_extracted:
            print(f"[EXISTS] Các file {expected_files} đã có sẵn trong {DATA_DIR}, bỏ qua.")
            continue

        archive_path = os.path.join(DATA_DIR, archive_name)
        print(f"\n[DOWNLOADING] {archive_name}...")
        try:
            with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=archive_name) as t:
                urllib.request.urlretrieve(url, filename=archive_path, reporthook=t.update_to)
        except Exception as e:
            print(f"[ERROR] Không thể tải {archive_name}: {e}")
            raise e

        print(f"[EXTRACTING] Đang giải nén {archive_name}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=DATA_DIR)

        # Xóa file nén tạm sau khi giải nén
        if os.path.exists(archive_path):
            os.remove(archive_path)

        print(f"  ✓ Đã giải nén thành công: {expected_files}")

    print("\n" + "=" * 60)
    print("TẢI & GIẢI NÉN TOÀN BỘ DỮ LIỆU IWSLT'15 THÀNH CÔNG!")
    print("=" * 60)

    # In thống kê số dòng từng file
    for f_name in ["train.en", "train.vi", "tst2012.en", "tst2012.vi", "tst2013.en", "tst2013.vi"]:
        p = os.path.join(DATA_DIR, f_name)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                num_lines = sum(1 for _ in f)
            print(f"  • {f_name:12s}: {num_lines:,} dòng ({os.path.getsize(p):,} bytes)")


if __name__ == "__main__":
    download_and_extract()
